"""Смешивание структурной и контекстной моделей.

Склеивание признаков в одну матрицу топит сотню контекстных столбцов в двух тысячах
структурных. Поэтому модели обучаются отдельно, а смешиваются их вероятности.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from experiments.context import build
from vacv.dataset import load
from vacv.pipeline import SENTINEL
from vacv.splits import fold_indices, scaffold_folds


def forest(seed: int = 0) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)


def out_of_fold(x: np.ndarray, labels: np.ndarray, folds) -> np.ndarray:
    probability = np.zeros(len(labels))
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        probability[validation_index] = forest().fit(x[train_index], labels[train_index]) \
            .predict_proba(x[validation_index])[:, 1]
    return probability


def main() -> None:
    data = load()
    structure, scaffolds, context, titles = build(data.train)
    y = data.y
    labels = (y == SENTINEL).astype(int)
    context_matrix = np.hstack([context["strain"].to_numpy(float), context["Culture"].to_numpy(float), titles])
    active_mean = y[labels == 0].mean()

    splits = {"scaffold": scaffold_folds(scaffolds, n_splits=5),
              "random": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)]}

    print(f"{'вес структуры':>14} {'AUC sc':>7} {'MSE sc':>12} {'AUC rn':>7} {'MSE rn':>12}")
    probabilities = {}
    for name, folds in splits.items():
        probabilities[name] = (out_of_fold(structure, labels, folds), out_of_fold(context_matrix, labels, folds))

    for weight in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        row = []
        for name in splits:
            structure_probability, context_probability = probabilities[name]
            blended = weight * structure_probability + (1 - weight) * context_probability
            prediction = blended * SENTINEL + (1 - blended) * active_mean
            row += [roc_auc_score(labels, blended), float(np.mean((prediction - y) ** 2))]
        print(f"{weight:14.1f} {row[0]:7.3f} {row[1]:12,.0f} {row[2]:7.3f} {row[3]:12,.0f}")


if __name__ == "__main__":
    main()
