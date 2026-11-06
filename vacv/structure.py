"""Структурная модель неактивности против VACV: ансамбль по блокам химических признаков.

Назначение — закрыть главное слабое место решения. В vacv/pipeline.py вес структурного
источника равен нулю, то есть прогноз держится на метаданных записи. Эта модель даёт
вероятность неактивности только из SMILES и подставляется в смесь вместо прежнего
структурного источника.

Устройство. Семь частей: шесть — ExtraTrees на одном блоке признаков каждая, седьмая —
ExtraTrees на отобранной конкатенации. Вероятности смешиваются весами, обратными MSE
соревнования, посчитанному внутренней 5-кратной стратифицированной CV на обучающей
выборке. Ridge-стекинг на том же N переобучается (docs/research.md, раздел 2.6), поэтому
второго уровня нет.

Числа и сравнение блоков — в docs/chemistry.md и experiments/chem_blocks.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

from .chemistry import RECOMMENDED, block_matrix
from .evaluate import SENTINEL, labels_of

PARTS: tuple[tuple[str, ...], ...] = (
    ("rdkit2d",),
    ("erg",),
    ("avalon",),
    ("maccs",),
    ("ecfp2",),
    ("physchem",),
    RECOMMENDED,
)


def _forest(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)


class StructureModel:
    """Вероятность того, что для соединения не будет измерена IC50 против VACV."""

    def __init__(self, seed: int = 0, n_splits: int = 5):
        self.seed = seed
        self.n_splits = n_splits

    def _matrices(self, frame: pd.DataFrame) -> list[np.ndarray]:
        cache = {name: block_matrix(frame["SMILES"], name) for name in set(sum(PARTS, ()))}
        return [np.hstack([cache[name] for name in part]) for part in PARTS]

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "StructureModel":
        labels = labels_of(y)
        matrices = self._matrices(frame)
        self.weights_ = self._weights(matrices, labels, y)
        self.models_ = [_forest(self.seed).fit(matrix, labels) for matrix in matrices]
        return self

    def _weights(self, matrices: list[np.ndarray], labels: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Веса, обратные OOF-MSE условного среднего. Средний SI активных — по фолду."""
        folds = list(StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
                     .split(labels, labels))
        errors = np.zeros(len(matrices))
        for train_index, validation_index in folds:
            level = y[train_index][labels[train_index] == 0].mean()
            for index, matrix in enumerate(matrices):
                model = _forest(self.seed).fit(matrix[train_index], labels[train_index])
                probability = model.predict_proba(matrix[validation_index])[:, 1]
                prediction = probability * SENTINEL + (1 - probability) * level
                errors[index] += np.sum((prediction - y[validation_index]) ** 2)
        weights = 1.0 / np.maximum(errors / len(labels), 1.0)
        return weights / weights.sum()

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        matrices = self._matrices(frame)
        return sum(weight * model.predict_proba(matrix)[:, 1]
                   for weight, model, matrix in zip(self.weights_, self.models_, matrices))
