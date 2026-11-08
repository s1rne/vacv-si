"""Сосед по номеру соединения внутри одной публикации.

В статье соединения пронумерованы подряд, и соседние номера обычно означают соседние
молекулы одной серии. Обучающие и тестовые соединения из одной статьи перемешаны,
поэтому для тестовой молекулы часто известна метка её ближайшего соседа по номеру.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.literature import attach
from vacv.pipeline import INACTIVE, SENTINEL, SelectivityModel, group_rate, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds

GROUP = ["strain", "Culture"]


def numbers(titles: pd.Series) -> np.ndarray:
    return pd.to_numeric(titles.astype(str).str.extract(r"^(\d+)", expand=False), errors="coerce").to_numpy(float)


def neighbour_features(train_frame: pd.DataFrame, labels: np.ndarray, frame: pd.DataFrame,
                       k: int = 3) -> np.ndarray:
    """Средняя метка k ближайших по номеру соединений той же публикации и расстояние до ближайшего."""
    train_numbers = numbers(train_frame["Title"])
    train_groups = list(map(tuple, train_frame[GROUP].values))
    prior = labels.mean()
    rows = []
    for number, group in zip(numbers(frame["Title"]), map(tuple, frame[GROUP].values)):
        mask = np.array([g == group for g in train_groups]) & np.isfinite(train_numbers)
        if not np.isfinite(number) or not mask.any():
            rows.append([prior, -1.0, 0.0])
            continue
        distance = np.abs(train_numbers[mask] - number)
        order = np.argsort(distance)[:k]
        rows.append([labels[mask][order].mean(), float(distance[order[0]]), float(len(order))])
    return np.array(rows)


def main() -> None:
    data = load()
    frame = attach(data.train)
    y = data.y
    labels = (y == INACTIVE).astype(int)
    splits = {"случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
              "остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5)}

    for name, folds in splits.items():
        neighbour = np.zeros(len(y))
        group = np.zeros(len(y))
        model_probability = np.zeros(len(y))
        for fold in range(len(folds)):
            train_index, validation_index = fold_indices(folds, fold)
            train_frame, valid_frame = frame.iloc[train_index], frame.iloc[validation_index]
            train_labels = labels[train_index]
            neighbour[validation_index] = neighbour_features(train_frame, train_labels, valid_frame)[:, 0]
            group[validation_index] = group_rate(train_frame, train_labels, valid_frame)
            model = SelectivityModel().fit(train_frame, y[train_index])
            model_probability[validation_index] = model.probability(valid_frame)

        active_mean = y[labels == 0].mean()
        def mse(p):
            return float(np.mean((p * SENTINEL + (1 - p) * active_mean - y) ** 2))
        print(f"\n{name}:")
        for label, probability in (("сосед по номеру", neighbour), ("доля по группе", group),
                                   ("текущая модель", model_probability)):
            print(f"  {label:20} AUC {roc_auc_score(labels, probability):.3f}  MSE {mse(probability):12,.0f}")
        for weight in (0.2, 0.3, 0.4):
            mixed = (1 - weight) * model_probability + weight * neighbour
            print(f"  модель + сосед {weight:.1f}    AUC {roc_auc_score(labels, mixed):.3f}  MSE {mse(mixed):12,.0f}")


if __name__ == "__main__":
    main()
