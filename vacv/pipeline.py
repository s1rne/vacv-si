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
from sklearn.ensemble import ExtraTreesClassifier

from .features import maccs, morgan_counts, parse, scaffold

INACTIVE = 9_999.0
SENTINEL = 10_000.0
STRUCTURE_WEIGHT = 0.2
CONTEXT_COLUMNS = ("strain", "Culture")


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
        return np.hstack(parts + [title_matrix(frame["Title"])])


class SelectivityModel:
    def __init__(self, seed: int = 0, structure_weight: float = STRUCTURE_WEIGHT):
        self.seed = seed
        self.structure_weight = structure_weight
        self.encoder = ContextEncoder()

    def _forest(self) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=self.seed)

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "SelectivityModel":
        inactive = (y >= INACTIVE).astype(int)
        self.active_mean_ = float(y[inactive == 0].mean())
        self.structure_ = self._forest().fit(structure_matrix(frame), inactive)
        self.encoder.fit(frame)
        self.context_ = self._forest().fit(self.encoder.transform(frame), inactive)
        return self

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        structure = self.structure_.predict_proba(structure_matrix(frame))[:, 1]
        context = self.context_.predict_proba(self.encoder.transform(frame))[:, 1]
        return self.structure_weight * structure + (1 - self.structure_weight) * context

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probability = self.probability(frame)
        return probability * SENTINEL + (1 - probability) * self.active_mean_
