"""Кросс-проверка с метрикой соревнования: MSE на исходной шкале SI."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.pipeline import INACTIVE, SelectivityModel, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def run(data, folds) -> tuple[float, float]:
    y = data.y
    prediction = np.zeros_like(y)
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        model = SelectivityModel().fit(data.train.iloc[train_index], y[train_index])
        prediction[validation_index] = model.predict(data.train.iloc[validation_index])
    inactive = (y >= INACTIVE).astype(int)
    auc = roc_auc_score(inactive, (prediction - y[inactive == 0].mean()))
    return auc, float(np.mean((prediction - y) ** 2))


def main() -> None:
    data = load()
    y = data.y
    splits = {
        "по остовам": scaffold_folds(scaffolds_of(data.train), n_splits=5),
        "случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
    }
    print(f"{'разбиение':12} {'AUC':>7} {'MSE':>13}")
    print(f"{'константа':12} {'—':>7} {np.mean((y.mean() - y) ** 2):13,.0f}")
    for name, folds in splits.items():
        auc, mse = run(data, folds)
        print(f"{name:12} {auc:7.3f} {mse:13,.0f}")


if __name__ == "__main__":
    main()
