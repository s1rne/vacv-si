"""Какой моделью был первый сабмит (публичный счёт 11.35M)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import frames, oracle
from vacv.pipeline import SelectivityModel

PUBLIC_FRACTION = 0.3


def main() -> None:
    train, test, y = frames()
    raw = pd.read_csv("data/raw/kaggle/test.csv")
    truth = oracle(raw)
    order = np.argsort(raw["index"].to_numpy())
    size = int(round(PUBLIC_FRACTION * len(truth)))

    def score(prediction: np.ndarray, name: str) -> None:
        squared = ((prediction - truth) ** 2)[order]
        print(f"{name:34} {squared[:size].mean():13,.0f} {squared.mean():13,.0f}")

    extra = [c for c in test.columns if c in ("matched", "literature_si")
             or c.startswith(("nn", "n_", "has_", "best_"))]
    plain, plain_test = train.drop(columns=extra), test.drop(columns=extra)
    print(f"{'кандидат':34} {'первые 30%':>13} {'полный тест':>13}")
    score(np.full(len(truth), y.mean()), "константа: среднее обучения")
    score(np.full(len(truth), np.median(y)), "константа: медиана обучения")
    for weights, name in (({"structure": 1.0, "context": 0.0, "grouped": 0.0}, "только структура"),
                          ({"structure": 0.0, "context": 1.0, "grouped": 0.0}, "только контекст")):
        prediction = np.mean([SelectivityModel(seed, weights=weights).fit(plain, y).predict(plain_test)
                              for seed in range(3)], axis=0)
        score(prediction, name)


if __name__ == "__main__":
    main()
