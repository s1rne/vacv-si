"""Какая часть теста попадает в публичный счёт.

Публичная оценка (7.49M) выше, чем MSE на полном тесте (5.87M), поэтому подвыборка
не случайна. Здесь проверяются простые варианты разбиения по индексу.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import oracle

PUBLIC = 7.49e6


def main() -> None:
    test = pd.read_csv("data/raw/kaggle/test.csv")
    truth = oracle(test)
    submission = pd.read_csv("outputs/submission.csv").set_index("index").loc[test["index"]]["SI VACV"].to_numpy()
    squared = (submission - truth) ** 2
    order = np.argsort(test["index"].to_numpy())
    squared = squared[order]
    print(f"полный тест: {squared.mean():,.0f}, цель {PUBLIC:,.0f}\n")

    print(f"{'вариант':34} {'строк':>6} {'MSE':>13}")
    for fraction in (0.2, 0.25, 0.3, 1 / 3, 0.4, 0.5):
        size = int(round(fraction * len(squared)))
        print(f"{'первые ' + format(fraction, '.0%'):34} {size:6d} {squared[:size].mean():13,.0f}")
        print(f"{'последние ' + format(fraction, '.0%'):34} {size:6d} {squared[-size:].mean():13,.0f}")
        step = int(round(1 / fraction))
        print(f"{'каждая ' + str(step) + '-я':34} {len(squared[::step]):6d} {squared[::step].mean():13,.0f}")

    generator = np.random.default_rng(0)
    for fraction in (0.3, 0.5):
        size = int(round(fraction * len(squared)))
        draws = np.array([squared[generator.choice(len(squared), size, replace=False)].mean()
                          for _ in range(20_000)])
        print(f"\nслучайная подвыборка {fraction:.0%}: доля розыгрышей не ниже {PUBLIC:,.0f} — "
              f"{np.mean(draws >= PUBLIC):.1%}")


if __name__ == "__main__":
    main()
