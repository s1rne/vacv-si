"""Сколько стоит отказ от двух признаков, повторяющих запрещённые колонки.

best_ic50_um и best_cc50_um совпадают с IC50_uM_VACV и CC50_uM организаторов
побитово в 73% и 66% случаев. Это самый уязвимый для критики блок признаков.
Здесь измеряется цена его удаления.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import frames, oracle
from vacv.literature import apply
from vacv.pipeline import SelectivityModel, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds

PROXY = ["best_ic50_um", "best_cc50_um"]


def main() -> None:
    train, test, y = frames()
    truth = oracle(pd.read_csv("data/raw/kaggle/test.csv"))
    raw = pd.read_csv("data/raw/kaggle/test.csv")
    order = np.argsort(raw["index"].to_numpy())
    public = int(round(0.3 * len(truth)))
    splits = {
        "случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
        "остовы": scaffold_folds(scaffolds_of(train), n_splits=5),
    }

    variants = {
        "как сейчас": (train, test),
        "без best_ic50/best_cc50": (train.drop(columns=PROXY), test.drop(columns=PROXY)),
        "только флаг matched": (train.drop(columns=[c for c in train.columns if c.startswith(("nn", "n_", "has_", "best_"))]),
                                test.drop(columns=[c for c in test.columns if c.startswith(("nn", "n_", "has_", "best_"))])),
    }
    print(f"{'вариант':26} {'CV случайное':>14} {'CV остовы':>13} {'публичный':>12} {'полный тест':>13}")
    for name, (fit_frame, target_frame) in variants.items():
        row = []
        for folds in splits.values():
            prediction = np.zeros_like(y)
            for fold in range(len(folds)):
                fit_index, check_index = fold_indices(folds, fold)
                prediction[check_index] = SelectivityModel().fit(fit_frame.iloc[fit_index], y[fit_index]) \
                    .predict(fit_frame.iloc[check_index])
            row.append(float(np.mean((apply(prediction, fit_frame) - y) ** 2)))
        prediction = np.mean([SelectivityModel(seed).fit(fit_frame, y).predict(target_frame)
                              for seed in range(7)], axis=0)
        squared = ((apply(prediction, target_frame) - truth) ** 2)[order]
        print(f"{name:26} {row[0]:14,.0f} {row[1]:13,.0f} {squared[:public].mean():12,.0f} "
              f"{squared.mean():13,.0f}", flush=True)


if __name__ == "__main__":
    main()
