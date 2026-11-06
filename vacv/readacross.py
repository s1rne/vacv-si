"""Read-across: признаки по ближайшим соседям обучающей выборки.

Перенос свойства с похожих соединений — принятый в регуляторной практике подход (OECD,
«read-across»), и химику он понятнее любого бустинга: соединение считается неактивным,
если ближайшие к нему известные соединения неактивны. Здесь это не отдельная модель, а
семь столбцов признаков, которые дальше идут в дерево вместе с дескрипторами.

Признаки зависят от меток обучающей части, поэтому считаются внутри фолда: соседи берутся
только из обучающих строк. Иначе метка контрольной молекулы попадёт в её собственный
признак.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "nn_inactive", "nn_active", "top3_inactive", "top3_active",
    "margin", "knn_vote", "nn_any",
]


def similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Матрица Танимото «запрос × обучение» для счётных отпечатков."""
    dot = query @ reference.T
    norm_query = np.einsum("ij,ij->i", query, query)[:, None]
    norm_reference = np.einsum("ij,ij->i", reference, reference)[None, :]
    denominator = norm_query + norm_reference - dot
    return np.divide(dot, denominator, out=np.zeros_like(dot, dtype=float), where=denominator > 0)


def _top(values: np.ndarray, count: int) -> np.ndarray:
    if values.shape[1] == 0:
        return np.zeros(values.shape[0])
    count = min(count, values.shape[1])
    return np.sort(values, axis=1)[:, -count:].mean(axis=1)


def features(query: np.ndarray, reference: np.ndarray, labels: np.ndarray, k: int = 5) -> np.ndarray:
    """Семь столбцов: близость к неактивным, к активным, их разность и голос k соседей."""
    matrix = similarity(query, reference)
    inactive, active = matrix[:, labels == 1], matrix[:, labels == 0]
    nn_inactive, nn_active = _top(inactive, 1), _top(active, 1)

    order = np.argsort(-matrix, axis=1)[:, :k]
    neighbour_similarity = np.take_along_axis(matrix, order, axis=1)
    neighbour_labels = labels[order]
    total = neighbour_similarity.sum(axis=1)
    vote = np.divide((neighbour_similarity * neighbour_labels).sum(axis=1), total,
                     out=np.full(len(matrix), labels.mean()), where=total > 0)

    return np.column_stack([nn_inactive, nn_active, _top(inactive, 3), _top(active, 3),
                            nn_inactive - nn_active, vote, matrix.max(axis=1)])
