"""Graph-based sentinel inference using unlabeled molecular/context geometry.

The graph may include the competition test FEATURES, never test targets. All
boundary labels and the prior come exclusively from the current training fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve

from .pipeline import INACTIVE, SENTINEL, structure_matrix


def graph_weights(frame, neighbours=6, cross_group=0.05):
    fingerprints = (structure_matrix(frame)[:, :2048] > 0).astype(float)
    product = fingerprints @ fingerprints.T
    denominator = fingerprints.sum(1)[:, None] + fingerprints.sum(1)[None, :] - product
    similarity = np.divide(product, denominator, out=np.zeros_like(product), where=denominator > 0)
    groups = frame[["strain", "Culture"]].fillna("").astype(str).to_numpy()
    same = (groups[:, None, :] == groups[None, :, :]).all(axis=-1)
    # Sparse local chemistry within a source group; weak bridges between sources.
    affinity = similarity ** 3 * np.where(same, 1.0, cross_group)
    np.fill_diagonal(affinity, 0.0)
    count = min(neighbours, len(frame) - 1)
    if count < 1:
        return affinity
    nearest = np.argpartition(affinity, -count, axis=1)[:, -count:]
    sparse = np.zeros_like(affinity)
    np.put_along_axis(sparse, nearest, np.take_along_axis(affinity, nearest, axis=1), axis=1)
    sparse = np.maximum(sparse, sparse.T)
    # Weak complete group edges avoid disconnecting singleton chemotypes.
    size = np.maximum(same.sum(axis=1), 1)
    sparse += 0.1 * same / np.sqrt(size[:, None] * size[None, :])
    np.fill_diagonal(sparse, 0.0)
    return sparse


class HarmonicSelectivityModel:
    def __init__(self, neighbours=6, regularization=0.03, cross_group=0.05,
                 evidence_strength=0.0):
        self.neighbours = neighbours
        self.regularization = regularization
        self.cross_group = cross_group
        self.evidence_strength = evidence_strength

    def fit_predict(self, train, y, validation, unlabeled=None):
        y = np.asarray(y, dtype=float)
        if len(y) != len(train) or not len(y) or not np.isfinite(y).all():
            raise ValueError("Training targets must be finite and aligned")
        frames = [train, validation]
        if unlabeled is not None:
            frames.append(unlabeled)
        # Deliberately whitelist inputs. Target and forbidden columns cannot enter.
        columns = ["SMILES", "strain", "Culture", "matched"]
        combined = pd.concat([f.reindex(columns=columns) for f in frames], ignore_index=True)
        weights = graph_weights(combined, self.neighbours, self.cross_group)
        labels = (y == INACTIVE).astype(float)
        n = len(train)
        degree = weights.sum(axis=1)
        unknown_weights = weights[n:, n:]
        penalty = np.full(len(combined) - n, self.regularization)
        rhs = weights[n:, :n] @ labels + penalty * labels.mean()
        if self.evidence_strength:
            # Shrink literature-matched nodes towards a fold-local, smoothed rate.
            matched_train = train["matched"].to_numpy() > 0
            rate = (labels[matched_train].sum() + labels.mean()) / (matched_train.sum() + 1)
            evidence = combined["matched"].fillna(0).to_numpy()[n:] > 0
            strength = self.evidence_strength * evidence
            penalty += strength
            rhs += strength * rate
        system = np.diag(degree[n:] + penalty) - unknown_weights
        probabilities = np.clip(solve(system, rhs, assume_a="pos"), 0, 1)[:len(validation)]
        active_mean = float(y[labels == 0].mean()) if (labels == 0).any() else 0.0
        prediction = probabilities * SENTINEL + (1 - probabilities) * active_mean
        return probabilities, prediction
