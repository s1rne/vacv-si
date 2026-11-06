"""Единый протокол оценки: одни и те же фолды и одна и та же метрика для всех моделей.

Метрика соревнования — MSE на исходной шкале SI. Около четверти обучающей выборки имеет
ровно SI = 10000, и это не измерение, а отметка «IC50 нет, соединение неактивно». Поэтому
прогноз собирается как условное среднее: вероятность неактивности умножается на 10000,
остаток — на оценку SI активных. Из-за этого любая модель сравнивается двумя числами:
ROC AUC классификатора неактивных и MSE готового прогноза.

Два разбиения. Случайное пятифолдовое — ориентир на лидерборд: тестовая выборка
соревнования не отделена от обучающей по химии. Разбиение по остовам Мурко — нижняя
граница на новых химических сериях.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from .dataset import load
from .features import parse, scaffold
from .splits import fold_indices, scaffold_folds

SENTINEL = 10_000.0
RANDOM_SEED = 0


def inactive_labels(y: np.ndarray) -> np.ndarray:
    """Метка неактивности — ровно значение-заглушка.

    Порог «не меньше 10000» ошибочно относит к неактивным строку с измеренным
    SI = 36066 (CC50 1082 при IC50 0.03).
    """
    return (np.asarray(y, dtype=np.float64) == SENTINEL).astype(int)


def scaffolds_of(smiles: pd.Series | list[str]) -> list[str]:
    return [scaffold(parse(value)) for value in smiles]


def folds_of(frame: pd.DataFrame, kind: str) -> list[np.ndarray]:
    if kind == "random":
        return [index for _, index in KFold(5, shuffle=True, random_state=RANDOM_SEED).split(frame)]
    if kind == "scaffold":
        return scaffold_folds(scaffolds_of(frame["SMILES"]), n_splits=5, seed=RANDOM_SEED)
    raise ValueError(f"неизвестное разбиение: {kind}")


@dataclass
class Task:
    """Обучающая выборка соревнования с готовыми фолдами."""

    frame: pd.DataFrame
    y: np.ndarray
    labels: np.ndarray
    folds: dict[str, list[np.ndarray]]

    @property
    def active_mean(self) -> float:
        return float(self.y[self.labels == 0].mean())

    def splits(self, kind: str):
        folds = self.folds[kind]
        for fold in range(len(folds)):
            yield fold_indices(folds, fold)


def task(frame: pd.DataFrame | None = None) -> Task:
    frame = load().train if frame is None else frame
    y = frame["SI VACV"].to_numpy(dtype=np.float64)
    return Task(frame=frame.reset_index(drop=True), y=y, labels=inactive_labels(y),
                folds={kind: folds_of(frame, kind) for kind in ("random", "scaffold")})


def conditional_mean(probability: np.ndarray, active: np.ndarray | float) -> np.ndarray:
    """Оптимальный по MSE прогноз: смесь заглушки и оценки SI активных."""
    probability = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    return probability * SENTINEL + (1.0 - probability) * np.asarray(active, dtype=np.float64)


def score(labels: np.ndarray, y: np.ndarray, probability: np.ndarray,
          active: np.ndarray | float) -> tuple[float, float]:
    """ROC AUC классификатора неактивных и MSE соревнования."""
    prediction = conditional_mean(probability, active)
    return float(roc_auc_score(labels, probability)), float(np.mean((prediction - y) ** 2))


@dataclass
class Result:
    """Строка итоговой таблицы: одна модель на одном разбиении."""

    model: str
    split: str
    auc: float
    mse: float
    seconds: float
    probability: np.ndarray

    def row(self) -> str:
        return f"{self.model:34} {self.split:9} {self.auc:6.3f} {self.mse:13,.0f} {self.seconds:8.1f}"


HEADER = f"{'модель':34} {'разбиение':9} {'AUC':>6} {'MSE':>13} {'сек':>8}"
