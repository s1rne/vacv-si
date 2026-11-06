"""Итоговая схема подстановки: два источника литературы и член неактивности."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.literature import apply, attach, attach_manual
from vacv.pipeline import SENTINEL, SelectivityModel, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def main() -> None:
    data = load()
    frame = attach_manual(attach(data.train), "train")
    y = data.y
    splits = {"случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
              "остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5)}

    for name, folds in splits.items():
        probability = np.zeros_like(y)
        active = np.zeros_like(y)
        for fold in range(len(folds)):
            train_index, validation_index = fold_indices(folds, fold)
            model = SelectivityModel().fit(frame.iloc[train_index], y[train_index])
            probability[validation_index] = model.probability(frame.iloc[validation_index])
            active[validation_index] = model.active_mean_
        plain = probability * SENTINEL + (1 - probability) * active

        without_manual = attach(data.train)
        print(f"\n{name}:")
        print(f"  только модель                        {np.mean((plain - y) ** 2):12,.0f}")
        print(f"  ChEMBL, жёсткая подстановка          "
              f"{np.mean((apply(plain, without_manual) - y) ** 2):12,.0f}")
        print(f"  ChEMBL, с членом неактивности        "
              f"{np.mean((apply(plain, without_manual, probability) - y) ** 2):12,.0f}")
        print(f"  ChEMBL + первоисточники              "
              f"{np.mean((apply(plain, frame, probability) - y) ** 2):12,.0f}")


if __name__ == "__main__":
    main()
