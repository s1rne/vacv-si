"""Сколько стоит отказ от признаков, повторяющих запрещённые колонки.

best_ic50_um и best_cc50_um совпадают с IC50_uM_VACV и CC50_uM организаторов
побитово в 73% и 66% случаев: это восстановленные из ChEMBL те же самые измерения.
Блок признаков присутствия включается целиком или не включается вовсе, поэтому
проверяются оба варианта отказа.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import frames, oracle
from vacv import literature, pipeline
from vacv.literature import apply
from vacv.pipeline import SelectivityModel, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds

FULL = list(literature.PRESENCE_FEATURES)
VARIANTS = {
    "как сейчас": FULL,
    "без best_ic50/best_cc50": [c for c in FULL if not c.startswith("best_")],
    "только nn*_tanimoto": [c for c in FULL if c.startswith("nn")],
    "без признаков присутствия": [],
}


def score(train: pd.DataFrame, y: np.ndarray, test: pd.DataFrame, truth: np.ndarray,
          splits: dict, order: np.ndarray, public: int) -> list[float]:
    out = []
    for folds in splits.values():
        prediction = np.zeros_like(y)
        for fold in range(len(folds)):
            fit_index, check_index = fold_indices(folds, fold)
            prediction[check_index] = SelectivityModel().fit(train.iloc[fit_index], y[fit_index]) \
                .predict(train.iloc[check_index])
        out.append(float(np.mean((apply(prediction, train) - y) ** 2)))
    prediction = np.mean([SelectivityModel(seed).fit(train, y).predict(test) for seed in range(7)], axis=0)
    squared = ((apply(prediction, test) - truth) ** 2)[order]
    return out + [float(squared[:public].mean()), float(squared.mean())]


def main() -> None:
    train, test, y = frames()
    raw = pd.read_csv("data/raw/kaggle/test.csv")
    truth = oracle(raw)
    order = np.argsort(raw["index"].to_numpy())
    public = int(round(0.3 * len(truth)))
    splits = {
        "случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
        "остовы": scaffold_folds(scaffolds_of(train), n_splits=5),
    }

    print(f"{'вариант':28} {'CV случайное':>14} {'CV остовы':>13} {'публичный':>12} {'полный тест':>13}")
    for name, features in VARIANTS.items():
        literature.PRESENCE_FEATURES[:] = features
        pipeline.PRESENCE_FEATURES = features
        values = score(train, y, test, truth, splits, order, public)
        print(f"{name:28} {values[0]:14,.0f} {values[1]:13,.0f} {values[2]:12,.0f} {values[3]:13,.0f}",
              flush=True)
    literature.PRESENCE_FEATURES[:] = FULL
    pipeline.PRESENCE_FEATURES = FULL


if __name__ == "__main__":
    main()
