"""Структурная модель неактивности против VACV: вероятность только из SMILES.

Назначение этой модели закрыть главное слабое место решения. В vacv/pipeline.py вес
структурного источника равен нулю, то есть прогноз держится на метаданных записи. Здесь
вероятность считается по одному строению молекулы, и её можно подставить в смесь.

Четыре блока признаков: дескрипторы RDKit без двух численно неустойчивых, ключи MACCS,
физхимия с правилами лекарственного подобия и фармакофорный сокращённый граф ErG. Модель
ExtraTrees: при 320 строках и 718 столбцах полностью случайные разрезы работают как
регуляризация, и она обходит CatBoost, LightGBM, XGBoost, SVM и KNN на обоих разбиениях.

Вероятности усредняются по пяти сидам: на 320 строках разброс одного леса по сидам сравним
с разницей между моделями, и усреднение его снимает почти бесплатно.

Ансамбль по блокам признаков с весами, обратными OOF-MSE, пробовали: он проигрывает одной
модели на конкатенации. Числа и сравнения в docs/chemistry.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from .chemistry import RECOMMENDED, features
from .evaluate import SENTINEL, labels_of


class StructureModel:
    """Вероятность того, что для соединения не будет измерена IC50 против VACV."""

    def __init__(self, seeds: int = 5, blocks: tuple[str, ...] = RECOMMENDED):
        self.seeds = seeds
        self.blocks = blocks

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "StructureModel":
        labels = labels_of(y)
        matrix = features(list(frame["SMILES"]), self.blocks)
        self.active_mean_ = float(y[labels == 0].mean())
        self.models_ = [
            ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)
            .fit(matrix, labels)
            for seed in range(self.seeds)
        ]
        return self

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = features(list(frame["SMILES"]), self.blocks)
        return np.mean([model.predict_proba(matrix)[:, 1] for model in self.models_], axis=0)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Условное среднее: тот же вид прогноза, что и у vacv.pipeline.SelectivityModel."""
        probability = self.probability(frame)
        return probability * SENTINEL + (1 - probability) * self.active_mean_
