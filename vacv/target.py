"""Работа с целевой переменной.

Ключевой факт от экспертов задачи: SI = 10000 не измерение, а признак того, что
IC50 отсутствует, то есть соединение против VACV неактивно. Вместо 10000 могло
стоять любое условно большое число. Поэтому такие записи цензурированы:
известно только, что истинный SI не меньше порога.
"""

from __future__ import annotations

import numpy as np

SENTINEL = 10_000.0


def is_censored(si: np.ndarray, sentinel: float = SENTINEL) -> np.ndarray:
    return np.asarray(si, dtype=np.float64) >= sentinel


def to_log(si: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """log10 SI: распределение SI тяжелохвостое, в логарифме ошибки сопоставимы по порядку."""
    return np.log10(np.clip(np.asarray(si, dtype=np.float64), floor, None))


def from_log(value: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(value, dtype=np.float64))


def censored_mse(prediction: np.ndarray, target: np.ndarray, censored: np.ndarray) -> np.ndarray:
    """Ошибка, не наказывающая завышение на цензурированных записях.

    Для обычной записи это квадрат разности. Для цензурированной штраф возникает,
    только если предсказание ниже порога: сверху истинное значение не ограничено.
    """
    residual = np.asarray(prediction) - np.asarray(target)
    residual = np.where(censored & (residual > 0), 0.0, residual)
    return residual**2
