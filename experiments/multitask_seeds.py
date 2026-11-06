"""Устойчив ли выигрыш от признаков с голов, или он в пределах шума одной выборки.

`multitask_stack.py` показывает, что четыре числа с голов внутри признаков источника
`grouped` дают около −0.5% MSE на случайном разбиении и −1.5% на остовах. По
docs/research.md (3.5) разрыв меньше 5–7% относительных на одной выборке неотличим
от шума, поэтому тот же замер повторяется на пяти затравках и печатается с разбросом.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.endpoints import load_corpus
from vacv.heads import HEAD_FEATURES, EndpointHeads
from vacv.literature import attach
from vacv.pipeline import (INACTIVE, SENTINEL, WEIGHTS, ContextEncoder, group_rate,
                           open_data_matrix, scaffolds_of, structure_matrix)
from vacv.splits import fold_indices, scaffold_folds

from experiments.multitask_stack import booster, forest

SEEDS = range(5)


def blend_probability(frame: pd.DataFrame, labels: np.ndarray, matrix: np.ndarray,
                      heads: np.ndarray | None, folds: list[np.ndarray], seed: int) -> np.ndarray:
    """Текущая смесь: 0.3 контекста плюс 0.7 структуры с долей неактивных в группе."""
    context = np.zeros(len(labels))
    grouped = np.zeros(len(labels))
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        train_frame, valid_frame = frame.iloc[train_index], frame.iloc[validation_index]
        train_labels = labels[train_index]

        encoder = ContextEncoder().fit(train_frame)
        context[validation_index] = forest(seed).fit(encoder.transform(train_frame), train_labels) \
            .predict_proba(encoder.transform(valid_frame))[:, 1]

        parts_train = [matrix[train_index], group_rate(train_frame, train_labels, train_frame)[:, None],
                       open_data_matrix(train_frame)]
        parts_valid = [matrix[validation_index], group_rate(train_frame, train_labels, valid_frame)[:, None],
                       open_data_matrix(valid_frame)]
        if heads is not None:
            parts_train.append(heads[train_index])
            parts_valid.append(heads[validation_index])
        grouped[validation_index] = booster(seed).fit(np.hstack(parts_train), train_labels) \
            .predict_proba(np.hstack(parts_valid))[:, 1]
    return WEIGHTS["context"] * context + WEIGHTS["grouped"] * grouped


def main() -> None:
    data = load()
    y = data.y
    labels = (y == INACTIVE).astype(int)
    frame = attach(data.train)
    matrix = structure_matrix(data.train)
    active_mean = float(y[labels == 0].mean())
    heads = EndpointHeads().fit(load_corpus()).predict(data.train)[HEAD_FEATURES].to_numpy(float)

    splits = {"случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
              "остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5)}

    print(f"{'разбиение':12} {'вариант':22} {'MSE':>22} {'AUC':>14}")
    for name, folds in splits.items():
        for label, extra in (("без голов", None), ("с головами", heads)):
            scores, aucs = [], []
            for seed in SEEDS:
                probability = blend_probability(frame, labels, matrix, extra, folds, seed)
                prediction = probability * SENTINEL + (1 - probability) * active_mean
                scores.append(float(np.mean((prediction - y) ** 2)))
                aucs.append(roc_auc_score(labels, probability))
            print(f"{name:12} {label:22} {np.mean(scores):13,.0f} ± {np.std(scores):6,.0f} "
                  f"{np.mean(aucs):8.3f} ± {np.std(aucs):.3f}")


if __name__ == "__main__":
    main()
