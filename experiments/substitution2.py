"""Как встроить литературное значение в условное среднее.

Жёсткая подстановка игнорирует возможность того, что соединение в выборке помечено
как неактивное. Правильнее подставлять литературное значение вместо константы для
активных, сохраняя член вероятности неактивности.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.literature import attach, lookup
from vacv.pipeline import SENTINEL, SelectivityModel, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds

CENSORED = ("lower_bound", "upper_bound", "undetermined")


def main() -> None:
    data = load()
    frame = attach(data.train)
    quality = data.train.merge(
        lookup()[["smiles", "si_quality"]].rename(columns={"smiles": "SMILES"}), on="SMILES", how="left"
    )["si_quality"].to_numpy(dtype=object)
    y = data.y
    splits = {"случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
              "остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5)}

    for name, folds in splits.items():
        probability = np.zeros_like(y)
        active_mean = np.zeros_like(y)
        for fold in range(len(folds)):
            train_index, validation_index = fold_indices(folds, fold)
            fitted = SelectivityModel().fit(frame.iloc[train_index], y[train_index])
            probability[validation_index] = fitted.probability(frame.iloc[validation_index])
            active_mean[validation_index] = fitted.active_mean_

        literature = np.clip(frame["literature_si"].to_numpy(dtype=float), 0, SENTINEL)
        found = np.isfinite(literature)
        censored = found & np.isin(quality, CENSORED)

        plain = probability * SENTINEL + (1 - probability) * active_mean
        hard = np.where(found, literature, plain)
        weighted = np.where(found, probability * SENTINEL + (1 - probability) * literature, plain)
        mixed = np.where(censored, literature, weighted)

        print(f"\n{name}:")
        for label, prediction in (("только модель", plain), ("жёсткая подстановка", hard),
                                  ("литература как значение активных", weighted),
                                  ("цензурированные подставить, прочие взвесить", mixed)):
            print(f"  {label:44} {np.mean((prediction - y) ** 2):12,.0f}")


if __name__ == "__main__":
    main()
