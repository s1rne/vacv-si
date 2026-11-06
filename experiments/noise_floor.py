"""Потолок качества: куда упираются любые модели на этих данных.

Опора одна: воспроизводимость публичных IC50 равна σ(pIC50) = 0.68 лог-единицы
(Kalliokoski 2013, 20 356 пар независимых измерений одной пары белок-лиганд;
docs/research.md, 4.5). Логарифм отношения складывает две такие ошибки, при независимости
σ(log10 SI) ≈ 0.96. Отсюда `RMSE_min = σ` и `R²_max = 1 − σ² / Var(y)`.

Второй блок считает то же для метрики соревнования. Там MSE на исходной шкале SI,
и условное среднее логнормального распределения при σ = 0.96 умножается
на exp((σ·ln10)²/2) ≈ 13, то есть шум попадает в метрику с огромным плечом.
Проверяется прямо: идеальный классификатор заглушки с константой для активных против
идеального классификатора с индивидуальным прогнозом активных на шумовом пределе.
"""

from __future__ import annotations

import numpy as np

from vacv.dataset import load
from vacv.endpoints import load_corpus
from vacv.pipeline import INACTIVE, SENTINEL

SIGMA_ENDPOINT = 0.68
SIGMA_RATIO = float(np.sqrt(2) * SIGMA_ENDPOINT)
REPEATS = 200


def main() -> None:
    corpus = load_corpus()
    data = load()
    y = data.y
    labels = y == INACTIVE
    active = ~labels & (y > 0)
    log_active = np.log10(y[active])

    rows = [
        ("log10 IC50, точные строки корпуса", corpus.loc[~corpus["ic50_censored"], "log_ic50"], SIGMA_ENDPOINT),
        ("log10 CC50, точные строки корпуса", corpus.loc[~corpus["cc50_censored"], "log_cc50"], SIGMA_ENDPOINT),
        ("log10 SI, строки measured корпуса", corpus.loc[corpus["si_quality"] == "measured", "log_si"], SIGMA_RATIO),
        ("log10 SI, активные строки соревнования", log_active, SIGMA_RATIO),
    ]
    print(f"{'задача':40} {'n':>5} {'Var':>6} {'R2_max':>7} {'RMSE_min':>9}")
    for name, series, sigma in rows:
        values = np.asarray(series, dtype=float)
        variance = float(values.var())
        print(f"{name:40} {values.size:5d} {variance:6.2f} {1 - sigma**2 / variance:7.2f} {sigma:9.2f}")

    print(f"\nметрика соревнования, {int(labels.sum())} заглушек из {len(y)}")
    constant = np.where(labels, SENTINEL, y[active].mean())
    print(f"  идеальный классификатор, константа для активных      {np.mean((constant - y) ** 2):12,.0f}")

    rng = np.random.default_rng(0)
    scores = []
    for _ in range(REPEATS):
        centre = log_active + rng.normal(0, SIGMA_RATIO, size=log_active.size)
        prediction = np.power(10.0, centre) * np.exp((SIGMA_RATIO * np.log(10)) ** 2 / 2)
        full = np.where(labels, SENTINEL, 0.0)
        full[active] = np.clip(prediction, 0, SENTINEL)
        scores.append(float(np.mean((full - y) ** 2)))
    print(f"  идеальный классификатор, активные на шумовом пределе {np.mean(scores):12,.0f}"
          f" ± {np.std(scores):,.0f}")
    print(f"  арифметическое среднее SI активных {y[active].mean():.1f}, "
          f"геометрическое {10 ** log_active.mean():.1f}")


if __name__ == "__main__":
    main()
