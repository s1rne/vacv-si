"""Общая подготовка данных для проверок. Кэширует вероятности модели на диске."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vacv.dataset import load
from vacv.literature import attach
from vacv.pipeline import SelectivityModel

CACHE = Path("analysis/cache")
SEEDS = 7


def oracle(frame: pd.DataFrame) -> np.ndarray:
    """Скрытая метка теста по правилу организаторов. Только для диагностики."""
    cc50 = frame["CC50_uM"].to_numpy(dtype=float)
    ic50 = frame["IC50_uM_VACV"].to_numpy(dtype=float)
    si = np.where(np.isnan(ic50), 10_000.0, cc50 / ic50)
    return np.where(np.isnan(cc50), 0.0, si)


def frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    data = load()
    return attach(data.train), attach(data.test), data.y


def test_truth() -> np.ndarray:
    return oracle(pd.read_csv("data/raw/kaggle/test.csv"))


def probabilities() -> tuple[np.ndarray, np.ndarray]:
    """Вероятность неактивности на тесте: средняя по сидам и матрица по сидам."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "test_probability.npy"
    if path.exists():
        matrix = np.load(path)
    else:
        train, test, y = frames()
        matrix = np.vstack([SelectivityModel(seed).fit(train, y).probability(test) for seed in range(SEEDS)])
        np.save(path, matrix)
    return matrix.mean(axis=0), matrix
