"""Оценка по метрике соревнования.

Два разбиения считаются всегда. Случайное — прокси лидерборда: тестовая выборка не
отделена от обучающей по химии, 98.5% тестовых строк попадают в группы strain+Culture
обучения, девять SMILES совпадают полностью. Разбиение по остовам Мурко — честная
граница для новой химии. Одно число без второго вводит в заблуждение.

MSE считается через то же условное среднее, что и vacv/pipeline.py: средний SI активных
берётся по обучающей части фолда, а не по всей выборке, иначе оценка подглядывает.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .chemistry import parse
from .features import scaffold
from .splits import fold_indices, scaffold_folds

SENTINEL = 10_000.0


def labels_of(y: np.ndarray) -> np.ndarray:
    """Метка неактивности: ровно значение-заглушка, а не «не меньше порога»."""
    return (np.asarray(y, dtype=float) == SENTINEL).astype(int)


def splits(frame: pd.DataFrame, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> dict[str, list[np.ndarray]]:
    scaffolds = [scaffold(parse(smiles)) for smiles in frame["SMILES"]]
    stratified = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return {
        "random": [index for _, index in stratified.split(y, labels_of(y))],
        "scaffold": scaffold_folds(scaffolds, n_splits=n_splits, seed=seed),
    }


def out_of_fold(predict: Callable[[np.ndarray, np.ndarray], np.ndarray], labels: np.ndarray,
                folds: list[np.ndarray]) -> np.ndarray:
    """predict(train_index, validation_index) -> вероятности неактивности на валидации."""
    probability = np.zeros(len(labels), dtype=float)
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        probability[validation_index] = predict(train_index, validation_index)
    return probability


def competition_mse(probability: np.ndarray, y: np.ndarray, folds: list[np.ndarray],
                    active: np.ndarray | None = None) -> float:
    """MSE на исходной шкале SI через условное среднее p·10000 + (1−p)·mean_active.

    `active` — необязательный прогноз SI для активных вместо константы фолда.
    """
    prediction = np.zeros(len(y), dtype=float)
    labels = labels_of(y)
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        train_active = y[train_index][labels[train_index] == 0]
        level = float(train_active.mean()) if active is None else active[validation_index]
        p = probability[validation_index]
        prediction[validation_index] = p * SENTINEL + (1 - p) * level
    return float(np.mean((prediction - y) ** 2))


def report(probability: dict[str, np.ndarray], y: np.ndarray,
           folds: dict[str, list[np.ndarray]]) -> dict[str, float]:
    """AUC и MSE на каждом разбиении для уже посчитанных OOF-вероятностей."""
    labels = labels_of(y)
    result = {}
    for name, p in probability.items():
        result[f"auc_{name}"] = float(roc_auc_score(labels, p))
        result[f"mse_{name}"] = competition_mse(p, y, folds[name])
    return result
