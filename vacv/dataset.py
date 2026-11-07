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
INACTIVE_INTERNAL = 10_000.0
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
    if not (train[TARGET] == INACTIVE_INTERNAL).any():
        # train_v2: организаторы пометили неактивные нулём. Внутри пайплайна неактивным
        # остаётся 10000, на выходе literature.to_zero_inactive возвращает нулевую метку.
        train[TARGET] = train[TARGET].mask(train[TARGET] == 0, INACTIVE_INTERNAL)
    for frame in (train, test):
        if frame[["SMILES", "strain", "Culture"]].isna().any().any():
            raise ValueError("пропуски в разрешённых полях")
    return Data(train, test)


def submission(index: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    index, prediction = np.asarray(index), np.asarray(prediction, dtype=float)
    if index.ndim != 1 or prediction.shape != index.shape or not len(index):
        raise ValueError("submission needs one prediction per index")
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("submission predictions must be finite and non-negative")
    if pd.isna(index).any() or pd.Index(index).has_duplicates:
        raise ValueError("submission indices must be present and unique")
    return pd.DataFrame({"index": index, TARGET: prediction})
