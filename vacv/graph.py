"""Распространение меток по графу похожести молекул.

Похожие по строению соединения чаще имеют одинаковый исход испытаний. Структуры
тестовых молекул известны заранее, поэтому метку «неактивно» можно переносить с
обучающих соединений на их близких соседей напрямую, не обучая на тесте.

Это перенос по графу, а не обучение на ответах: используются только метки обучающей
части, а рёбра строятся по отпечаткам Моргана, то есть по строению.
"""

from __future__ import annotations

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .features import parse


def fingerprints(smiles: list[str], radius: int = 2, size: int = 2048):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
    return [generator.GetFingerprint(parse(s)) for s in smiles]


def similarity(query: list, reference: list) -> np.ndarray:
    """Матрица сходства Танимото, строки — запросы, столбцы — опорные соединения."""
    return np.vstack([DataStructs.BulkTanimotoSimilarity(f, reference) for f in query])


def propagate(matrix: np.ndarray, labels: np.ndarray, neighbours: int = 10,
              sharpness: float = 8.0, prior: float | None = None) -> np.ndarray:
    """Взвешенная доля неактивных среди ближайших соседей.

    Вес соседа равен сходству в степени [sharpness]: так далёкие соседи почти не влияют.
    Если подходящих соседей нет, возвращается общая доля неактивных.
    """
    prior = labels.mean() if prior is None else prior
    order = np.argsort(-matrix, axis=1)[:, :neighbours]
    rows = np.arange(len(matrix))[:, None]
    weights = np.power(np.clip(matrix[rows, order], 0.0, 1.0), sharpness)
    total = weights.sum(axis=1)
    value = (weights * labels[order]).sum(axis=1)
    return np.where(total > 1e-9, value / np.maximum(total, 1e-9), prior)


def features(train_smiles: list[str], train_labels: np.ndarray, smiles: list[str],
             neighbours: tuple[int, ...] = (1, 5, 15)) -> np.ndarray:
    """Набор признаков: доля неактивных среди соседей разного радиуса и само сходство."""
    reference = fingerprints(train_smiles)
    matrix = similarity(fingerprints(smiles), reference)
    columns = [propagate(matrix, train_labels, k) for k in neighbours]
    columns.append(matrix.max(axis=1))
    columns.append(np.sort(matrix, axis=1)[:, -5:].mean(axis=1))
    return np.column_stack(columns)
