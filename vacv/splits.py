"""Схемы разбиения.

Случайное разбиение по молекулам завышает качество: серии аналогов из одной статьи
попадают и в обучение, и в контроль. Разбиение по остову Мурко показывает, что будет
на новых химических сериях.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def scaffold_folds(scaffolds: list[str], n_splits: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Фолды, в которых молекулы одного остова не разделяются. Группы кладутся в самый пустой фолд."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(scaffolds):
        groups[key or f"_{index}"].append(index)

    order = sorted(groups.values(), key=len, reverse=True)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for members in order:
        folds[int(np.argmin([len(f) for f in folds]))].extend(members)
    return [np.array(sorted(f)) for f in folds]


def fold_indices(folds: list[np.ndarray], fold: int) -> tuple[np.ndarray, np.ndarray]:
    validation = folds[fold]
    train = np.concatenate([f for i, f in enumerate(folds) if i != fold])
    return np.sort(train), validation
