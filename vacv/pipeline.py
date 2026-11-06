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

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier

from .features import maccs, morgan_counts, parse, scaffold
from .literature import PRESENCE_FEATURES

SENTINEL = 10_000.0
# Метка неактивности — ровно значение-заглушка. Порог «не меньше 9999» ошибочно относил
# к неактивным строку с настоящим SI = 36066 (CC50 1082 при IC50 0.03).
INACTIVE = SENTINEL
CONTEXT_COLUMNS = ("strain", "Culture")
GROUP_COLUMNS = ["strain", "Culture"]
WEIGHTS = {"structure": 0.0, "context": 0.3, "grouped": 0.7}
SMOOTHING = 5.0


def structure_matrix(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for smiles in frame["SMILES"]:
        mol = parse(smiles)
        if mol is None:
            raise ValueError(f"не разбирается SMILES: {smiles}")
        rows.append(np.concatenate([morgan_counts(mol), maccs(mol)]))
    return np.vstack(rows)


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


class SelectivityModel:
    """Три источника вероятности неактивности, смешанные с фиксированными весами.

    structure  строение молекулы;
    context    штамм, культура и обозначение соединения;
    grouped    строение плюс доля неактивных в группе одним признаком.

    Веса подобраны на случайном разбиении: тестовая выборка не отделена от обучающей
    по химии, 98.5% групп общие, девять молекул совпадают полностью.
    """

    def __init__(self, seed: int = 0, weights: dict[str, float] | None = None):
        self.seed = seed
        self.weights = weights or WEIGHTS
        self.encoder = ContextEncoder()

    def _forest(self) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=self.seed)

    def _booster(self) -> LGBMClassifier:
        return LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                              colsample_bytree=0.3, reg_lambda=5.0, verbose=-1, random_state=self.seed)

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "SelectivityModel":
        labels = (y == INACTIVE).astype(int)
        self.active_mean_ = float(y[labels == 0].mean())
        self.train_frame_ = frame
        self.labels_ = labels
        matrix = structure_matrix(frame)

        if self.weights["structure"]:
            self.structure_ = self._forest().fit(matrix, labels)
        if self.weights["context"]:
            self.encoder.fit(frame)
            self.context_ = self._forest().fit(self.encoder.transform(frame), labels)
        if self.weights["grouped"]:
            self.grouped_ = self._booster().fit(self._grouped_features(frame, matrix), labels)
        return self

    def _grouped_features(self, frame: pd.DataFrame, matrix: np.ndarray) -> np.ndarray:
        rate = group_rate(self.train_frame_, self.labels_, frame)[:, None]
        return np.hstack([matrix, rate, open_data_matrix(frame)])

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = structure_matrix(frame)
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
