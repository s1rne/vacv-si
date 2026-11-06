"""Source-specific chemistry effects with penalized kernel logistic regression."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .kernel import KernelSelectivityModel
from .pipeline import INACTIVE, SENTINEL


def source_kernel(left, right):
    a, ag, ae = left
    b, bg, be = right
    dot = a @ b.T
    denominator = a.sum(1)[:, None] + b.sum(1)[None, :] - dot
    similarity = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0) ** 2
    group = (ag[:, None, :] == bg[None, :, :]).all(axis=-1)
    culture = ag[:, None, 1] == bg[None, :, 1]
    evidence = ae[:, None] == be[None, :]
    # Source random intercept + source-specific chemistry + shared chemistry.
    return similarity + 1.5 * group + 2 * similarity * group + 0.5 * culture + 0.5 * group * evidence


class SourceKernelClassifier:
    def __init__(self, C=3.0):
        self.C = C

    def fit_predict(self, train, y, validation, unlabeled=None):
        y = np.asarray(y)
        labels = (y == INACTIVE).astype(int)
        if len(np.unique(labels)) == 1:
            p = np.full(len(validation), labels[0], dtype=float)
        else:
            train_features = KernelSelectivityModel._inputs(train)
            valid_features = KernelSelectivityModel._inputs(validation)
            matrix = source_kernel(train_features, train_features)
            values, vectors = np.linalg.eigh(matrix)
            keep = values > 1e-9
            values, vectors = values[keep], vectors[:, keep]
            fit_features = vectors * np.sqrt(values)
            valid_features = source_kernel(valid_features, train_features) @ (vectors / np.sqrt(values))
            estimator = LogisticRegression(C=self.C, solver="lbfgs", max_iter=2000)
            estimator.fit(fit_features, labels)
            p = estimator.predict_proba(valid_features)[:, 1]
        active_mean = float(y[labels == 0].mean()) if (labels == 0).any() else 0.0
        return p, p * SENTINEL + (1 - p) * active_mean
