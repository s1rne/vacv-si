"""Ordered categorical boosting candidate; no hand-made target encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .pipeline import (CONTEXT_COLUMNS, INACTIVE, SENTINEL, open_data_matrix,
                       structure_matrix, title_matrix)


class BoostedSelectivityModel:
    def __init__(self, seed=0, depth=4, molecular=True):
        self.seed = seed
        self.depth = depth
        self.molecular = molecular

    def _features(self, frame):
        parts = [title_matrix(frame["Title"]), open_data_matrix(frame)]
        if self.molecular:
            parts.append(structure_matrix(frame, with_descriptors=True))
        result = pd.DataFrame(np.hstack(parts))
        result.columns = [f"feature_{i}" for i in range(result.shape[1])]
        for name in CONTEXT_COLUMNS:
            result[name] = frame[name].fillna("").astype(str).to_numpy()
        result["group"] = result["strain"] + "|" + result["Culture"]
        return result

    def fit(self, frame, y):
        y = np.asarray(y, dtype=float)
        labels = (y == INACTIVE).astype(int)
        self.active_mean_ = float(y[labels == 0].mean()) if (labels == 0).any() else 0.0
        self.constant_ = float(labels[0]) if len(np.unique(labels)) == 1 else None
        if self.constant_ is None:
            self.model_ = CatBoostClassifier(
                iterations=1000, depth=self.depth, learning_rate=0.035,
                l2_leaf_reg=5, loss_function="Logloss", boosting_type="Ordered",
                random_seed=self.seed, verbose=False, allow_writing_files=False,
                thread_count=4,
            ).fit(self._features(frame), labels, cat_features=[*CONTEXT_COLUMNS, "group"])
        return self

    def probability(self, frame):
        if self.constant_ is not None:
            return np.full(len(frame), self.constant_)
        return self.model_.predict_proba(self._features(frame))[:, 1]

    def predict(self, frame):
        p = self.probability(frame)
        return p * SENTINEL + (1 - p) * self.active_mean_
