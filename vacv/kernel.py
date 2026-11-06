"""Small-data kernel model combining molecular and publication similarity."""

from __future__ import annotations

import numpy as np

from .pipeline import INACTIVE, SENTINEL, structure_matrix


class KernelSelectivityModel:
    def __init__(self, seed=0, alpha=0.3):
        self.alpha = alpha

    @staticmethod
    def _inputs(frame):
        fingerprint = (structure_matrix(frame)[:, :2048] > 0).astype(float)
        group = np.array(list(map(tuple, frame[["strain", "Culture"]].to_numpy())), dtype=object)
        evidence = frame["matched"].to_numpy(float) if "matched" in frame else np.zeros(len(frame))
        return fingerprint, group, evidence

    @staticmethod
    def _kernel(left, right):
        a, ag, ae = left
        b, bg, be = right
        dot = a @ b.T
        denominator = a.sum(1)[:, None] + b.sum(1)[None, :] - dot
        similarity = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
        same_group = (ag[:, None, :] == bg[None, :, :]).all(axis=2)
        same_evidence = ae[:, None] == be[None, :]
        # Each summand is a positive-semidefinite kernel (including products).
        return similarity ** 2 + 2 * same_group + same_group * same_evidence

    def fit(self, frame, y):
        y = np.asarray(y)
        labels = (y == INACTIVE).astype(float)
        self.active_mean_ = float(y[labels == 0].mean()) if (labels == 0).any() else 0.0
        self.prior_ = labels.mean()
        self.inputs_ = self._inputs(frame)
        matrix = self._kernel(self.inputs_, self.inputs_)
        self.coefficients_ = np.linalg.solve(matrix + self.alpha * np.eye(len(frame)), labels - self.prior_)
        return self

    def probability(self, frame):
        return np.clip(self.prior_ + self._kernel(self._inputs(frame), self.inputs_) @ self.coefficients_, 0, 1)

    def predict(self, frame):
        p = self.probability(frame)
        return p * SENTINEL + (1 - p) * self.active_mean_
