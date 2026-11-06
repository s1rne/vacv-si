"""Сколько шума в публичном лидерборде.

Публичный счёт считается на подвыборке теста. Распределение построчных квадратов
ошибки крайне неравномерно: несколько строк дают проценты суммы. Отсюда разброс
публичной оценки при одной и той же модели.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import oracle


def main() -> None:
    test = pd.read_csv("data/raw/kaggle/test.csv")
    truth = oracle(test)
    submission = pd.read_csv("outputs/submission.csv").set_index("index").loc[test["index"]]["SI VACV"].to_numpy()
    squared = (submission - truth) ** 2
    full = squared.mean()
    print(f"MSE на полном тесте {full:,.0f}")
    share = np.sort(squared)[::-1] / squared.sum()
    print(f"вклад 10 худших строк {share[:10].sum():.1%}, 30 худших {share[:30].sum():.1%}, "
          f"половины строк {share[:len(share) // 2].sum():.1%}")

    generator = np.random.default_rng(0)
    print(f"\n{'доля теста':>11} {'строк':>6} {'медиана MSE':>13} {'интервал 5–95%':>26}")
    for fraction in (0.2, 0.3, 0.4, 0.5):
        size = int(round(fraction * len(truth)))
        draws = np.array([squared[generator.choice(len(squared), size, replace=False)].mean()
                          for _ in range(20_000)])
        print(f"{fraction:11.0%} {size:6d} {np.median(draws):13,.0f} "
              f"{np.percentile(draws, 5):11,.0f} – {np.percentile(draws, 95):<11,.0f}")

    print("\nразница двух моделей, которую публичная оценка различает:")
    for fraction in (0.3, 0.5):
        size = int(round(fraction * len(truth)))
        draws = np.array([squared[generator.choice(len(squared), size, replace=False)].mean()
                          for _ in range(20_000)])
        print(f"  при доле {fraction:.0%} стандартная ошибка оценки {draws.std():,.0f}; "
              f"разрыв меньше {2 * draws.std():,.0f} статистически не виден")


if __name__ == "__main__":
    main()
