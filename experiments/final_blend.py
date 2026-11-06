"""Подбор смеси трёх источников вероятности на случайном разбиении.

Тестовая выборка соревнования не отделена от обучающей по химии: 98.5% групп общие,
девять молекул совпадают полностью. Поэтому ориентир — случайное разбиение, а оценка
по остовам приводится как нижняя граница на новых химических сериях.
"""

from __future__ import annotations

import itertools

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold

from experiments.within_group import target_encode
from vacv.dataset import load
from vacv.pipeline import SENTINEL, ContextEncoder, structure_matrix, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def forest(seed=0):
    return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)


def booster(seed=0):
    return LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                          colsample_bytree=0.3, reg_lambda=5.0, verbose=-1, random_state=seed)


def sources(data, labels, matrix, folds) -> dict[str, np.ndarray]:
    out = {name: np.zeros(len(labels)) for name in ("structure", "context", "grouped")}
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        train_frame, valid_frame = data.train.iloc[train_index], data.train.iloc[validation_index]
        train_labels = labels[train_index]
        out["structure"][validation_index] = forest().fit(matrix[train_index], train_labels) \
            .predict_proba(matrix[validation_index])[:, 1]
        encoder = ContextEncoder().fit(train_frame)
        out["context"][validation_index] = forest().fit(encoder.transform(train_frame), train_labels) \
            .predict_proba(encoder.transform(valid_frame))[:, 1]
        rate_train = target_encode(train_frame, train_labels, train_frame)[:, None]
        rate_valid = target_encode(train_frame, train_labels, valid_frame)[:, None]
        out["grouped"][validation_index] = booster().fit(
            np.hstack([matrix[train_index], rate_train]), train_labels) \
            .predict_proba(np.hstack([matrix[validation_index], rate_valid]))[:, 1]
    return out


def main() -> None:
    data = load()
    y = data.y
    labels = (y == SENTINEL).astype(int)
    matrix = structure_matrix(data.train)
    active_mean = y[labels == 0].mean()

    splits = {"случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
              "остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5)}
    probability = {name: sources(data, labels, matrix, folds) for name, folds in splits.items()}

    def mse(p):
        return float(np.mean((p * SENTINEL + (1 - p) * active_mean - y) ** 2))

    grid = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=3) if abs(sum(w) - 1) < 1e-9]
    scored = sorted(((mse(sum(weight * probability["случайное"][name]
                              for weight, name in zip(w, ("structure", "context", "grouped")))), w) for w in grid))
    print(f"{'структура':>10} {'контекст':>9} {'группа':>7} {'MSE случ.':>12} {'MSE остовы':>12}")
    for score, w in scored[:6]:
        on_scaffold = mse(sum(weight * probability["остовы"][name]
                              for weight, name in zip(w, ("structure", "context", "grouped"))))
        print(f"{w[0]:10.1f} {w[1]:9.1f} {w[2]:7.1f} {score:12,.0f} {on_scaffold:12,.0f}")
    print("\nтекущая модель (0.2/0.8/0):")
    for name, table in probability.items():
        print(f"  {name:10} {mse(0.2 * table['structure'] + 0.8 * table['context']):12,.0f}")


if __name__ == "__main__":
    main()
