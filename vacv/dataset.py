"""Загрузка данных соревнования.

Правила соревнования запрещают использовать колонки CC50_uM и IC50_uM_VACV для
построения модели: целевая переменная вычисляется из них арифметически. Поэтому
загрузчик их не отдаёт. Колонки strain и Culture использовать разрешено, это
подтвердили организаторы в чате задачи.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FORBIDDEN = ("CC50_uM", "IC50_uM_VACV")
TARGET = "SI VACV"
RAW = Path("data/raw/kaggle")


@dataclass
class Data:
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def y(self) -> np.ndarray:
        return self.train[TARGET].to_numpy(dtype=np.float64)


def load(folder: Path = RAW) -> Data:
    train = pd.read_csv(folder / "train.csv").drop(columns=list(FORBIDDEN))
    test = pd.read_csv(folder / "test.csv").drop(columns=list(FORBIDDEN))
    for frame in (train, test):
        if frame[["SMILES", "strain", "Culture"]].isna().any().any():
            raise ValueError("пропуски в разрешённых полях")
    return Data(train, test)


def submission(index: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"index": index, TARGET: prediction})
