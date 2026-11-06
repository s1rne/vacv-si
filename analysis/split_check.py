"""Проверка гипотезы о публичном разбиении на всех поколениях модели."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import frames, oracle
from cv_vs_lb import GENERATIONS, fit_predict
from vacv.literature import apply

PUBLIC_FRACTION = 0.3


def main() -> None:
    train, test, y = frames()
    raw = pd.read_csv("data/raw/kaggle/test.csv")
    truth = oracle(raw)
    order = np.argsort(raw["index"].to_numpy())
    size = int(round(PUBLIC_FRACTION * len(truth)))

    rows = []
    for generation in GENERATIONS:
        prediction = np.mean([fit_predict(train, y, test, generation, seed) for seed in range(7)], axis=0)
        if generation.literature:
            prediction = apply(prediction, test)
        squared = ((prediction - truth) ** 2)[order]
        rows.append({
            "поколение": generation.name,
            "LB факт": generation.lb,
            "первые 30%": squared[:size].mean(),
            "полный тест": squared.mean(),
            "остальные 70%": squared[size:].mean(),
        })
        print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    table["невязка"] = table["первые 30%"] - table["LB факт"]
    table.to_csv("analysis/split_check.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print(f"\nмаксимальная относительная невязка "
          f"{(table['невязка'].abs() / table['LB факт']).max():.2%}")


if __name__ == "__main__":
    main()
