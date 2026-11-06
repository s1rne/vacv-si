"""Калибровка вероятности и значение для активных.

Прогноз p·10000 + (1−p)·m чувствителен к смещению p: деревья склонны стягивать
вероятности к середине. Проверяются степенное преобразование p, изотоническая
калибровка и подбор константы m.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.pipeline import INACTIVE, SENTINEL, ContextEncoder, structure_matrix, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def forest(seed=0):
    return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)


def blended_oof(data, labels, folds) -> np.ndarray:
    matrix = structure_matrix(data.train)
    probability = np.zeros(len(labels))
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        structure = forest().fit(matrix[train_index], labels[train_index]) \
            .predict_proba(matrix[validation_index])[:, 1]
        encoder = ContextEncoder().fit(data.train.iloc[train_index])
        context = forest().fit(encoder.transform(data.train.iloc[train_index]), labels[train_index]) \
            .predict_proba(encoder.transform(data.train.iloc[validation_index]))[:, 1]
        probability[validation_index] = 0.2 * structure + 0.8 * context
    return probability


def mse(probability, y, active_value) -> float:
    prediction = probability * SENTINEL + (1 - probability) * active_value
    return float(np.mean((prediction - y) ** 2))


def main() -> None:
    data = load()
    y = data.y
    labels = (y >= INACTIVE).astype(int)
    active_mean = y[labels == 0].mean()
    splits = {"остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5),
              "случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)]}

    for name, folds in splits.items():
        probability = blended_oof(data, labels, folds)
        print(f"\n{name}: AUC {roc_auc_score(labels, probability):.3f}")
        print(f"  без правок, m={active_mean:.0f}            {mse(probability, y, active_mean):,.0f}")

        best = min(((power, mse(np.clip(probability, 0, 1) ** power, y, active_mean))
                    for power in np.arange(0.5, 2.01, 0.1)), key=lambda item: item[1])
        print(f"  степень p^{best[0]:.1f}                   {best[1]:,.0f}")

        isotonic = IsotonicRegression(out_of_bounds="clip").fit(probability, labels)
        print(f"  изотоническая калибровка        {mse(isotonic.predict(probability), y, active_mean):,.0f}"
              f"   (оптимистично: подогнана на тех же данных)")

        grid = min(((value, mse(probability, y, value)) for value in np.arange(0, 1001, 25)), key=lambda i: i[1])
        print(f"  подбор m={grid[0]:.0f}                     {grid[1]:,.0f}")

        both = min(((power, value, mse(np.clip(probability, 0, 1) ** power, y, value))
                    for power in np.arange(0.5, 2.01, 0.1) for value in np.arange(0, 1001, 50)),
                   key=lambda item: item[2])
        print(f"  вместе: p^{both[0]:.1f}, m={both[1]:.0f}          {both[2]:,.0f}")


if __name__ == "__main__":
    main()
