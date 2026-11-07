"""Модель предсказания индекса селективности.

Метрика соревнования — MSE на исходной шкале SI, и около четверти выборки имеет ровно
SI = 10000. По словам экспертов это не измерение, а отметка «IC50 отсутствует,
соединение против VACV неактивно». Поэтому ошибку определяет прежде всего то, отнесено
ли соединение к неактивным: промах здесь стоит порядка 10⁸, а ошибка в разы на активном
соединении — порядка 10². Оптимальный по MSE прогноз — условное среднее, и оно собирается
из вероятности неактивности и среднего SI среди активных.

Вероятность неактивности даёт смесь двух моделей. Первая смотрит на строение молекулы,
вторая — на контекст записи: вирусный штамм, культуру клеток и обозначение соединения.
Контекст косвенно указывает на публикацию-источник, а внутри одной публикации серия
соединений обычно либо активна, либо нет, поэтому он предсказывает сильнее структуры.
Склеивать признаки в одну матрицу нельзя: сотня контекстных столбцов тонет в двух тысячах
структурных, поэтому смешиваются именно вероятности.

Колонки CC50_uM и IC50_uM_VACV не используются: правила соревнования это запрещают.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold

from .features import descriptors, maccs, morgan_counts, parse, scaffold
from .literature import PRESENCE_FEATURES

# Заглушка «IC50 отсутствует». Сравнение ровно на равенство, а не по порогу: одна
# обучающая строка имеет настоящий SI = 36066 (CC50 1082 при IC50 0.03), и порог
# «не меньше 9999» отнёс бы измерение к неактивным.
SENTINEL = 10_000.0
# Метка неактивности — ровно значение-заглушка. Порог «не меньше 9999» ошибочно относил
# к неактивным строку с настоящим SI = 36066.667.
INACTIVE = SENTINEL
CONTEXT_COLUMNS = ("strain", "Culture")
GROUP_COLUMNS = ["strain", "Culture"]
WEIGHTS = {"structure": 0.0, "context": 0.5, "grouped": 0.5}
SMOOTHING = 5.0


@lru_cache(maxsize=4096)
def _structure_row(smiles: str, with_descriptors: bool) -> np.ndarray:
    mol = parse(smiles)
    # Invalid molecules still have context/evidence. Never drop submission rows.
    if mol is None:
        width = 2048 + 167 + (len(descriptors(parse("C"))) if with_descriptors else 0)
        return np.zeros(width, dtype=np.float32)
    parts = [morgan_counts(mol), maccs(mol)]
    if with_descriptors:
        parts.append(descriptors(mol))
    return np.concatenate(parts)


def structure_matrix(frame: pd.DataFrame, with_descriptors: bool = False) -> np.ndarray:
    return np.vstack([_structure_row(smiles, with_descriptors) for smiles in frame["SMILES"]])


def scaffolds_of(frame: pd.DataFrame) -> list[str]:
    return [scaffold(parse(smiles)) for smiles in frame["SMILES"]]


def title_matrix(titles: pd.Series) -> np.ndarray:
    """Обозначение соединения из статьи: длина, цифры, ведущий номер."""
    text = titles.astype(str)
    return np.column_stack([
        text.str.len(),
        text.str.count(r"\d"),
        text.str.match(r"^\d+$").astype(float),
        text.str.contains("-").astype(float),
        text.str.extract(r"^(\d+)", expand=False).astype(float).fillna(-1),
    ])


def open_data_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Признаки присутствия соединения в открытых базах. Пропуски заменяются на −1."""
    columns = ["matched", *PRESENCE_FEATURES]
    if not set(columns) <= set(frame.columns):
        return np.zeros((len(frame), 0))
    return frame[columns].to_numpy(dtype=float, na_value=-1.0)


class ContextEncoder:
    """Категории штамма и культуры фиксируются по обучающей выборке."""

    def fit(self, frame: pd.DataFrame) -> "ContextEncoder":
        self.columns_ = {name: pd.get_dummies(frame[name].astype("category"), prefix=name).columns
                         for name in CONTEXT_COLUMNS}
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts = [pd.get_dummies(frame[name].astype("category"), prefix=name)
                 .reindex(columns=self.columns_[name], fill_value=0).to_numpy(float)
                 for name in CONTEXT_COLUMNS]
        return np.hstack(parts + [title_matrix(frame["Title"]), open_data_matrix(frame)])


def group_rate(train_frame: pd.DataFrame, labels: np.ndarray, frame: pd.DataFrame,
               smoothing: float = SMOOTHING) -> np.ndarray:
    """Доля неактивных в группе strain+Culture, сглаженная к общему среднему.

    Группа отвечает публикации-источнику и покрывает 98.5% тестовых строк.
    Сглаживание не даёт маленьким группам давать крайние оценки.
    """
    prior = labels.mean()
    stats = pd.DataFrame({"group": list(map(tuple, train_frame[GROUP_COLUMNS].values)), "label": labels}) \
        .groupby("group")["label"].agg(["size", "mean"])
    encoded = (stats["size"] * stats["mean"] + smoothing * prior) / (stats["size"] + smoothing)
    return np.array([encoded.get(key, prior) for key in map(tuple, frame[GROUP_COLUMNS].values)])


def crossfit_group_rate(frame: pd.DataFrame, labels: np.ndarray, seed: int = 0,
                        smoothing: float = SMOOTHING) -> np.ndarray:
    """Training rates use only other inner folds, including the global prior."""
    if len(frame) < 2:
        return np.full(len(frame), 0.5)
    rates = np.empty(len(frame))
    for train, valid in KFold(min(5, len(frame)), shuffle=True, random_state=seed).split(frame):
        rates[valid] = group_rate(frame.iloc[train], labels[train], frame.iloc[valid], smoothing)
    return rates


class SelectivityModel:
    """Три источника вероятности неактивности, смешанные с фиксированными весами.

    structure  строение молекулы;
    context    штамм, культура и обозначение соединения;
    grouped    строение плюс доля неактивных в группе одним признаком.

    Веса подобраны на случайном разбиении: тестовая выборка не отделена от обучающей
    по химии, 98.5% групп общие, девять молекул совпадают полностью.
    """

    def __init__(self, seed: int = 0, weights: dict[str, float] | None = None,
                 group_encoding: str = "legacy", with_descriptors: bool = False,
                 smoothing: float = SMOOTHING, forest_leaf: int = 2,
                 forest_features: str | float = "sqrt"):
        self.seed = seed
        self.weights = dict(WEIGHTS if weights is None else weights)
        if (set(self.weights) != set(WEIGHTS)
                or not all(np.isfinite(v) and v >= 0 for v in self.weights.values())
                or not np.isclose(sum(self.weights.values()), 1.0)):
            raise ValueError("weights must be non-negative structure/context/grouped weights summing to 1")
        if group_encoding not in {"legacy", "crossfit"}:
            raise ValueError("group_encoding must be legacy or crossfit")
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be positive and finite")
        self.group_encoding = group_encoding
        self.with_descriptors = with_descriptors
        self.smoothing = smoothing
        self.forest_leaf = forest_leaf
        self.forest_features = forest_features
        self.encoder = ContextEncoder()

    def _forest(self) -> ExtraTreesClassifier:
        # n_jobs=1: при n_jobs=-1 порядок суммирования по потокам плавает, и вероятности
        # расходятся на 2e-16, из-за чего файл ответа не воспроизводится побайтово.
        return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=self.forest_leaf,
                                    max_features=self.forest_features, n_jobs=1, random_state=self.seed)

    def _booster(self) -> LGBMClassifier:
        return LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                              colsample_bytree=0.3, reg_lambda=5.0, verbose=-1, random_state=self.seed)

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "SelectivityModel":
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not len(y) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("y must contain one finite non-negative target per row")
        labels = (y == INACTIVE).astype(int)
        self.active_mean_ = float(y[labels == 0].mean()) if (labels == 0).any() else 0.0
        self.constant_probability_ = float(labels[0]) if len(np.unique(labels)) == 1 else None
        self.train_frame_ = frame.copy()
        self.labels_ = labels
        if self.constant_probability_ is not None:
            return self
        matrix = structure_matrix(frame, self.with_descriptors)

        if self.weights["structure"]:
            self.structure_ = self._forest().fit(matrix, labels)
        if self.weights["context"]:
            self.encoder.fit(frame)
            self.context_ = self._forest().fit(self.encoder.transform(frame), labels)
        if self.weights["grouped"]:
            rate = (crossfit_group_rate(frame, labels, self.seed, self.smoothing)
                    if self.group_encoding == "crossfit" else None)
            self.grouped_ = self._booster().fit(self._grouped_features(frame, matrix, rate), labels)
        return self

    def _grouped_features(self, frame: pd.DataFrame, matrix: np.ndarray,
                          rate: np.ndarray | None = None) -> np.ndarray:
        if rate is None:
            rate = group_rate(self.train_frame_, self.labels_, frame, self.smoothing)
        return np.hstack([matrix, rate[:, None], open_data_matrix(frame)])

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        if self.constant_probability_ is not None:
            return np.full(len(frame), self.constant_probability_)
        matrix = structure_matrix(frame, self.with_descriptors)
        total = np.zeros(len(frame))
        if self.weights["structure"]:
            total += self.weights["structure"] * self.structure_.predict_proba(matrix)[:, 1]
        if self.weights["context"]:
            total += self.weights["context"] * self.context_.predict_proba(self.encoder.transform(frame))[:, 1]
        if self.weights["grouped"]:
            total += self.weights["grouped"] * self.grouped_.predict_proba(
                self._grouped_features(frame, matrix))[:, 1]
        return total

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probability = self.probability(frame)
        return probability * SENTINEL + (1 - probability) * self.active_mean_
