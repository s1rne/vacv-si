"""SI соревнования, собранный из голов, обученных на внешнем корпусе.

Метрика соревнования считает MSE на исходной шкале SI, и при четверти строк с заглушкой
10000 оптимальный прогноз даёт условное среднее:

    E[SI] = p · 10000 + (1 − p) · E[SI | активно].

Это та же формула, что в `vacv/pipeline.py`. Отличие в источнике её частей: вероятность
неактивности p приходит с головы, обученной на метке `ic50_relation == '>'` внешнего
корпуса, а E[SI | активно] собирается из разности предсказанных log10 CC50 и log10 IC50.

Две поправки обязательны и обе однопараметрические, то есть не могут подогнаться под шум:

1. Сдвиг базовой частоты. В корпусе неактивны 59% соединений (там крупные HTS-скрины),
   в выборке соревнования 25%. Без сдвига логита p систематически завышена, а промах
   по заглушке стоит порядка 10⁸ за строку.
2. Сдвиг шкалы отношения. Корпус собран по другим протоколам, и предсказанный log10 SI
   смещён относительно печатного SI соревнования. Смещение снимается линейной
   калибровкой по активным строкам обучающего фолда.

Переход из логарифма в исходную шкалу делается с поправкой на логнормальность:
10^m даёт геометрическое среднее, а метрика требует арифметического.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .heads import EndpointHeads
from .pipeline import INACTIVE, SENTINEL

LOG10 = np.log(10.0)


def logit(probability: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), floor, 1 - floor)
    return np.log(p / (1 - p))


def expit(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def shift_prior(probability: np.ndarray, source: float, target: float) -> np.ndarray:
    """Перенос вероятности с базовой частоты корпуса на базовую частоту выборки."""
    return expit(logit(probability) - logit(np.array(source)) + logit(np.array(target)))


class MultitaskModel:
    """Головы корпуса плюс тонкая калибровка по обучающему фолду соревнования.

    `active='heads'` берёт средний SI активных из разности голов, `active='constant'`
    из среднего по фолду. Второй режим нужен как контроль: эксперимент `grid.py` показал,
    что на этой метрике модель активных почти ничего не решает.
    """

    def __init__(self, heads: EndpointHeads, active: str = "heads", shift: bool = True):
        self.heads = heads
        self.active = active
        self.shift = shift

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "MultitaskModel":
        labels = (y == INACTIVE).astype(int)
        predicted = self.heads.predict(frame)
        self.target_prior_ = float(labels.mean())
        self.active_mean_ = float(y[labels == 0].mean())
        self.floor_ = float(np.min(y[labels == 0]))

        active = labels == 0
        log_si = predicted.loc[active, "log_si"].to_numpy(float)
        log_y = np.log10(np.clip(y[active], 1e-3, None))
        design = np.column_stack([np.ones_like(log_si), log_si])
        self.coefficients_ = np.linalg.lstsq(design, log_y, rcond=None)[0]
        residual = log_y - design @ self.coefficients_
        self.residual_sd_ = float(residual.std())
        return self

    def _active_mean(self, predicted: pd.DataFrame) -> np.ndarray:
        if self.active == "constant":
            return np.full(len(predicted), self.active_mean_)
        log_si = predicted["log_si"].to_numpy(float)
        centre = self.coefficients_[0] + self.coefficients_[1] * log_si
        mean = np.power(10.0, centre) * np.exp((self.residual_sd_ * LOG10) ** 2 / 2)
        return np.clip(mean, self.floor_, SENTINEL)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        predicted = self.heads.predict(frame)
        probability = predicted["p_inactive"].to_numpy(float)
        if self.shift:
            probability = shift_prior(probability, self.heads.prior_, self.target_prior_)
        return probability * SENTINEL + (1 - probability) * self._active_mean(predicted)
