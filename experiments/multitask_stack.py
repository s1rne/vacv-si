"""Стоит ли добавлять головы корпуса в текущий блендинг.

Текущая модель смешивает три вероятности неактивности: по строению, по контексту записи
(штамм, культура, обозначение) и по строению вместе с долей неактивных в группе. Головы
корпуса дают четвёртый источник той же вероятности и, отдельно, четыре числовых признака
(`log_ic50`, `log_cc50`, `log_si`, `p_inactive`), которые можно просто дописать к матрице
признаков.

Проверяются оба способа. Веса смеси берутся обратными OOF-MSE, как рекомендует docs/research.md,
плюс полный перебор сетки с шагом 0.1 для верхней границы выигрыша.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.endpoints import load_corpus
from vacv.heads import HEAD_FEATURES, EndpointHeads
from vacv.literature import attach
from vacv.multitask import shift_prior
from vacv.pipeline import (INACTIVE, SENTINEL, ContextEncoder, group_rate, open_data_matrix,
                           scaffolds_of, structure_matrix)
from vacv.splits import fold_indices, scaffold_folds

NAMES = ("structure", "context", "grouped", "heads")


def forest(seed: int = 0) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)


def booster(seed: int = 0) -> LGBMClassifier:
    return LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                          colsample_bytree=0.3, reg_lambda=5.0, verbose=-1, random_state=seed)


def sources(frame: pd.DataFrame, labels: np.ndarray, matrix: np.ndarray,
            head_frame: pd.DataFrame, prior: float, folds: list[np.ndarray]) -> dict[str, np.ndarray]:
    """OOF-вероятности четырёх источников плюс вариант с головами внутри признаков."""
    out = {name: np.zeros(len(labels)) for name in (*NAMES, "grouped_heads")}
    heads = head_frame[HEAD_FEATURES].to_numpy(float)
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        train_frame, valid_frame = frame.iloc[train_index], frame.iloc[validation_index]
        train_labels = labels[train_index]

        out["structure"][validation_index] = forest().fit(matrix[train_index], train_labels) \
            .predict_proba(matrix[validation_index])[:, 1]
        encoder = ContextEncoder().fit(train_frame)
        out["context"][validation_index] = forest().fit(encoder.transform(train_frame), train_labels) \
            .predict_proba(encoder.transform(valid_frame))[:, 1]

        rate_train = group_rate(train_frame, train_labels, train_frame)[:, None]
        rate_valid = group_rate(train_frame, train_labels, valid_frame)[:, None]
        grouped_train = np.hstack([matrix[train_index], rate_train, open_data_matrix(train_frame)])
        grouped_valid = np.hstack([matrix[validation_index], rate_valid, open_data_matrix(valid_frame)])
        out["grouped"][validation_index] = booster().fit(grouped_train, train_labels) \
            .predict_proba(grouped_valid)[:, 1]
        out["grouped_heads"][validation_index] = booster().fit(
            np.hstack([grouped_train, heads[train_index]]), train_labels) \
            .predict_proba(np.hstack([grouped_valid, heads[validation_index]]))[:, 1]

        out["heads"][validation_index] = shift_prior(
            head_frame["p_inactive"].to_numpy(float)[validation_index], prior, train_labels.mean())
    return out


def main() -> None:
    data = load()
    y = data.y
    labels = (y == INACTIVE).astype(int)
    frame = attach(data.train)
    matrix = structure_matrix(data.train)
    active_mean = float(y[labels == 0].mean())
    heads = EndpointHeads().fit(load_corpus())
    head_frame = heads.predict(data.train)

    splits = {"случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
              "остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5)}
    probability = {name: sources(frame, labels, matrix, head_frame, heads.prior_, folds)
                   for name, folds in splits.items()}

    def mse(p: np.ndarray) -> float:
        return float(np.mean((p * SENTINEL + (1 - p) * active_mean - y) ** 2))

    print(f"{'источник':16} " + " ".join(f"{name:>12} {'AUC':>6}" for name in splits))
    for name in (*NAMES, "grouped_heads"):
        cells = []
        for split in splits:
            p = probability[split][name]
            cells.append(f"{mse(p):12,.0f} {roc_auc_score(labels, p):6.3f}")
        print(f"{name:16} " + " ".join(cells))

    current = {"structure": 0.0, "context": 0.3, "grouped": 0.7, "heads": 0.0}
    inverse = {name: 1.0 / mse(probability["случайное"][name]) for name in NAMES}
    total = sum(inverse.values())
    inverse = {name: value / total for name, value in inverse.items()}

    grid = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=4) if abs(sum(w) - 1) < 1e-9]

    def combine(split: str, weights: dict[str, float], key: str = "grouped") -> np.ndarray:
        return sum(weights[name] * probability[split][key if name == "grouped" else name]
                   for name in NAMES)

    best = min(grid, key=lambda w: mse(combine("случайное", dict(zip(NAMES, w)))))
    print("\nсмеси")
    print(f"{'вариант':34} {'стр':>4} {'кон':>4} {'груп':>5} {'гол':>4} "
          + " ".join(f"{name:>14}" for name in splits))
    for label, weights in (("текущая 0/0.3/0.7", current),
                           ("обратные MSE", inverse),
                           ("лучшая на сетке", dict(zip(NAMES, best)))):
        cells = " ".join(f"{mse(combine(split, weights)):14,.0f}" for split in splits)
        print(f"{label:34} {weights['structure']:4.1f} {weights['context']:4.1f} "
              f"{weights['grouped']:5.1f} {weights['heads']:4.1f} {cells}")
    cells = " ".join(f"{mse(combine(split, current, key='grouped_heads')):14,.0f}" for split in splits)
    print(f"{'текущая, головы в признаках':34} {current['structure']:4.1f} "
          f"{current['context']:4.1f} {current['grouped']:5.1f} {'в призн.':>8} {cells}")


if __name__ == "__main__":
    main()
