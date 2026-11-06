"""Correct-and-Smooth-inspired correction from strictly inner-OOF residuals.

This adapts fixed residual diffusion, not the complete C&S benchmark pipeline.
Reference: https://arxiv.org/abs/2010.13993. Residual sign here is label - probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.model_selection import KFold

from .inference import predict_ensemble
from .literature import PRESENCE_FEATURES, apply
from .pipeline import INACTIVE, SENTINEL
from .transductive import graph_weights

INPUT_COLUMNS = ["SMILES", "strain", "Culture", "Title", "matched", "literature_si", *PRESENCE_FEATURES]


def safe_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Neither targets nor competition assay columns can reach an estimator."""
    return frame[[column for column in INPUT_COLUMNS if column in frame]].copy()


def crossfit_probabilities(frame: pd.DataFrame, y: np.ndarray, seeds: int = 1,
                           n_splits: int = 4, split_seed: int = 2718) -> np.ndarray:
    """Every probability excludes its row's label and all outer-validation labels."""
    y = np.asarray(y, dtype=float)
    if y.shape != (len(frame),) or not np.isfinite(y).all() or len(y) < n_splits:
        raise ValueError("Finite aligned targets and at least n_splits rows are required")
    if n_splits < 2:
        raise ValueError("Cross-fitting needs at least two folds")
    features = safe_features(frame)
    result = np.full(len(frame), np.nan)
    for train, valid in KFold(n_splits, shuffle=True, random_state=split_seed).split(frame):
        result[valid] = predict_ensemble(features.iloc[train], y[train], features.iloc[valid],
                                         seeds=seeds, profile="hybrid")[0]
    if not np.isfinite(result).all():
        raise ValueError("Incomplete cross-fitted probabilities")
    return result


def propagate_residuals(weights: np.ndarray, residuals: np.ndarray,
                        mode: str = "diffusion", regularization: float = .03) -> np.ndarray:
    """Known first-n residuals are fixed boundaries; unknown prior correction is zero.

    One-hop removes unknown-to-unknown edges from the same graph. Diffusion solves
    (D_uu + lambda I - W_uu) delta_u = W_ul residual_l exactly.
    """
    weights = np.asarray(weights, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    if (weights.ndim != 2 or weights.shape[0] != weights.shape[1] or
            n < 1 or n >= len(weights) or residuals.shape != (n,) or
            not np.isfinite(weights).all() or not np.isfinite(residuals).all() or
            (weights < 0).any() or not np.allclose(weights, weights.T)):
        raise ValueError("Symmetric finite nonnegative graph and aligned residual vector required")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Regularization must be strictly positive")
    boundary = weights[n:, :n]
    rhs = boundary @ residuals
    if mode == "onehop":
        return rhs / (boundary.sum(axis=1) + regularization)
    if mode != "diffusion":
        raise ValueError(f"Unknown propagation mode: {mode}")
    system = np.diag(weights[n:].sum(axis=1) + regularization) - weights[n:, n:]
    return solve(system, rhs, assume_a="pos")


def correct_probability(probability: np.ndarray, delta: np.ndarray, beta: float,
                        uncertainty_gate: bool = False) -> np.ndarray:
    """Optional fixed confidence gate shrinks corrections at near-certain base predictions."""
    probability, delta = np.asarray(probability), np.asarray(delta)
    if (probability.ndim != 1 or probability.shape != delta.shape or
            not np.isfinite(probability).all() or not np.isfinite(delta).all() or
            (probability < 0).any() or (probability > 1).any()):
        raise ValueError("Aligned finite probabilities and corrections required")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    gate = 4 * probability * (1 - probability) if uncertainty_gate else 1.
    return np.clip(probability + beta * gate * delta, 0., 1.)


class GraphResidualCorrector:
    """Strong hybrid plus graph propagation of its inner-fold prediction errors."""

    def __init__(self, seeds: int = 1, inner_splits: int = 4, inner_seed: int = 2718,
                 neighbours: int = 6, cross_group: float = .05, regularization: float = .03):
        self.seeds, self.inner_splits, self.inner_seed = seeds, inner_splits, inner_seed
        self.neighbours, self.cross_group, self.regularization = neighbours, cross_group, regularization

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> "GraphResidualCorrector":
        self.train_ = safe_features(train)
        self.y_ = np.asarray(y, dtype=float).copy()
        self.inner_probability_ = crossfit_probabilities(self.train_, self.y_, self.seeds,
                                                        self.inner_splits, self.inner_seed)
        self.residuals_ = (self.y_ == INACTIVE).astype(float) - self.inner_probability_
        active = self.y_[self.y_ != INACTIVE]
        self.active_mean_ = float(active.mean()) if len(active) else 0.
        return self

    def components(self, query: pd.DataFrame, unlabeled: pd.DataFrame | None = None) -> dict:
        query = safe_features(query)
        probability, raw, final = predict_ensemble(self.train_, self.y_, query,
                                                   seeds=self.seeds, profile="hybrid")
        frames = [self.train_, query]
        if unlabeled is not None:
            frames.append(safe_features(unlabeled))
        graph = graph_weights(pd.concat(frames, ignore_index=True), self.neighbours, self.cross_group)
        deltas = {mode: propagate_residuals(graph, self.residuals_, mode, self.regularization)[:len(query)]
                  for mode in ["onehop", "diffusion"]}
        return {"probability": probability, "raw": raw, "final": final, "deltas": deltas}

    def predictions(self, query: pd.DataFrame, components: dict, beta: float, mode: str,
                    uncertainty_gate: bool = False):
        probability = correct_probability(components["probability"], components["deltas"][mode],
                                          beta, uncertainty_gate)
        raw = probability * SENTINEL + (1 - probability) * self.active_mean_
        return probability, raw, apply(raw, query)
