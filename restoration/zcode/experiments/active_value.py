"""Нужно ли разное среднее SI активных для найденных и ненайденных в литературе соединений."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.literature import apply, attach
from vacv.pipeline import INACTIVE, SENTINEL, SelectivityModel, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def main() -> None:
    data = load()
    frame = attach(data.train)
    y = data.y
    splits = {"случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
              "остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5)}

    for name, folds in splits.items():
        single = np.zeros_like(y)
        split_mean = np.zeros_like(y)
        for fold in range(len(folds)):
            train_index, validation_index = fold_indices(folds, fold)
            model = SelectivityModel().fit(frame.iloc[train_index], y[train_index])
            probability = model.probability(frame.iloc[validation_index])
            train_y = y[train_index]
            active = train_y != INACTIVE
            matched_train = frame["matched"].to_numpy()[train_index] > 0
            matched_valid = frame["matched"].to_numpy()[validation_index] > 0
            global_mean = train_y[active].mean()
            by_match = np.where(matched_valid,
                                train_y[active & matched_train].mean(),
                                train_y[active & ~matched_train].mean())
            single[validation_index] = probability * SENTINEL + (1 - probability) * global_mean
            split_mean[validation_index] = probability * SENTINEL + (1 - probability) * by_match

        for label, prediction in (("одно среднее", single), ("раздельные средние", split_mean)):
            combined = apply(prediction, frame)
            print(f"{name:10} {label:20} {np.mean((combined - y) ** 2):12,.0f}")


if __name__ == "__main__":
    main()
