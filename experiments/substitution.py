"""Жёсткая подстановка литературного значения или смесь с прогнозом модели.

У восстановленных значений разное качество: измеренное, нижняя граница, верхняя
граница, неопределённое. Проверяется, выгодно ли доверять каждому классу полностью.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.literature import attach, lookup
from vacv.pipeline import SelectivityModel, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


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
        model = np.zeros_like(y)
        for fold in range(len(folds)):
            train_index, validation_index = fold_indices(folds, fold)
            fitted = SelectivityModel().fit(frame.iloc[train_index], y[train_index])
            model[validation_index] = fitted.predict(frame.iloc[validation_index])

        literature = frame["literature_si"].to_numpy(dtype=float)
        found = np.isfinite(literature)
        print(f"\n{name}: без подстановки {np.mean((model - y) ** 2):12,.0f}")
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            mixed = model.copy()
            mixed[found] = (1 - weight) * model[found] + weight * np.clip(literature[found], 0, 10_000)
            print(f"  доля литературы {weight:4.2f}   {np.mean((mixed - y) ** 2):12,.0f}")

        print("  по классам качества (полная подстановка против модели):")
        for cls in ("measured", "lower_bound", "upper_bound", "undetermined"):
            mask = found & (quality == cls)
            if not mask.any():
                continue
            model_error = np.mean((model[mask] - y[mask]) ** 2)
            literature_error = np.mean((np.clip(literature[mask], 0, 10_000) - y[mask]) ** 2)
            print(f"    {cls:14} n={mask.sum():3d}  модель {model_error:11,.0f}  литература {literature_error:11,.0f}")


if __name__ == "__main__":
    main()
