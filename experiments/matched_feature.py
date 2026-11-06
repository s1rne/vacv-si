"""Что даёт признак «молекула найдена в открытых базах»."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.literature import apply, attach
from vacv.pipeline import SelectivityModel, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def run(frame, y, folds) -> np.ndarray:
    prediction = np.zeros_like(y)
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        model = SelectivityModel().fit(frame.iloc[train_index], y[train_index])
        prediction[validation_index] = model.predict(frame.iloc[validation_index])
    return prediction


def main() -> None:
    data = load()
    y = data.y
    plain = data.train
    enriched = attach(data.train)
    splits = {"случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
              "остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5)}
    for name, folds in splits.items():
        base = run(plain, y, folds)
        with_flag = run(enriched, y, folds)
        combined = apply(with_flag, enriched)
        print(f"{name:10} без признака {np.mean((base - y) ** 2):12,.0f}   "
              f"с признаком {np.mean((with_flag - y) ** 2):12,.0f}   "
              f"плюс подстановка {np.mean((combined - y) ** 2):12,.0f}")


if __name__ == "__main__":
    main()
