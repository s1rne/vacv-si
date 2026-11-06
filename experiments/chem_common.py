"""Общая обвязка химических экспериментов: данные, кэш блоков признаков, прогон по фолдам."""

from __future__ import annotations

import numpy as np

from experiments.chem_zoo import CLASSIFIERS
from vacv.chemistry import block_matrix
from vacv.dataset import load
from vacv.evaluate import labels_of, out_of_fold, splits

DATA = load()
Y = DATA.y
LABELS = labels_of(Y)
FOLDS = splits(DATA.train, Y)
SMILES = list(DATA.train["SMILES"])

_cache: dict[str, np.ndarray] = {}


def block(name: str) -> np.ndarray:
    if name not in _cache:
        _cache[name] = block_matrix(SMILES, name)
    return _cache[name]


def matrix(names) -> np.ndarray:
    return np.hstack([block(name) for name in names])


def classifier_oof(name: str, x: np.ndarray, folds: list[np.ndarray],
                   labels: np.ndarray | None = None) -> np.ndarray:
    labels = LABELS if labels is None else labels
    build = CLASSIFIERS[name]

    def predict(train_index: np.ndarray, validation_index: np.ndarray) -> np.ndarray:
        model = build().fit(x[train_index], labels[train_index])
        return model.predict_proba(x[validation_index])[:, 1]

    return out_of_fold(predict, labels, folds)
