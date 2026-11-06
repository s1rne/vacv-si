"""Честная проверка калибровки: калибратор учится только на обучающей части фолда."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.pipeline import INACTIVE, SENTINEL, ContextEncoder, structure_matrix, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def forest(seed=0):
    return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)


def blend(train_frame, train_labels, test_frame, matrix, train_index, test_index) -> tuple[np.ndarray, np.ndarray]:
    structure_model = forest().fit(matrix[train_index], train_labels)
    encoder = ContextEncoder().fit(train_frame)
    context_model = forest().fit(encoder.transform(train_frame), train_labels)

    def probability(frame, index):
        return 0.2 * structure_model.predict_proba(matrix[index])[:, 1] + \
            0.8 * context_model.predict_proba(encoder.transform(frame))[:, 1]

    return probability(train_frame, train_index), probability(test_frame, test_index)


def main() -> None:
    data = load()
    y = data.y
    labels = (y >= INACTIVE).astype(int)
    matrix = structure_matrix(data.train)
    splits = {"остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5),
              "случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)]}

    for name, folds in splits.items():
        plain = np.zeros(len(y))
        isotonic_prediction = np.zeros(len(y))
        platt = np.zeros(len(y))
        for fold in range(len(folds)):
            train_index, validation_index = fold_indices(folds, fold)
            inner = scaffold_folds([f"{i}" for i in range(len(train_index))], n_splits=4) if False else None
            train_probability, valid_probability = blend(
                data.train.iloc[train_index], labels[train_index],
                data.train.iloc[validation_index], matrix, train_index, validation_index)
            plain[validation_index] = valid_probability
            isotonic = IsotonicRegression(out_of_bounds="clip").fit(train_probability, labels[train_index])
            isotonic_prediction[validation_index] = isotonic.predict(valid_probability)
            scaler = LogisticRegression().fit(train_probability.reshape(-1, 1), labels[train_index])
            platt[validation_index] = scaler.predict_proba(valid_probability.reshape(-1, 1))[:, 1]

        active_mean = y[labels == 0].mean()
        def mse(probability):
            return float(np.mean((probability * SENTINEL + (1 - probability) * active_mean - y) ** 2))
        print(f"{name:10} без калибровки {mse(plain):12,.0f}   изотоническая {mse(isotonic_prediction):12,.0f}"
              f"   Платт {mse(platt):12,.0f}")


if __name__ == "__main__":
    main()
