"""Shared inference path for validation and submission."""

from __future__ import annotations

import numpy as np

from .literature import apply
from .pipeline import SENTINEL, SelectivityModel

PROFILES = {
    "baseline": {},
    "crossfit": {"group_encoding": "crossfit"},
    "descriptors": {"group_encoding": "crossfit", "with_descriptors": True},
    "context": {"weights": {"structure": 0.0, "context": 1.0, "grouped": 0.0}},
    "catboost": {"depth": 4},
    "catboost6": {"depth": 6},
    "catcontext": {"depth": 4, "molecular": False},
    "kernel": {"alpha": 0.3},
    "kernel1": {"alpha": 1.0},
    "context_full": {"weights": {"structure": 0.0, "context": 1.0, "grouped": 0.0},
                     "forest_leaf": 1, "forest_features": 1.0},
    "context_half": {"weights": {"structure": 0.0, "context": 1.0, "grouped": 0.0},
                     "forest_leaf": 1, "forest_features": 0.5},
    "context_leaf1": {"weights": {"structure": 0.0, "context": 1.0, "grouped": 0.0},
                      "forest_leaf": 1},
    "blend": {},
    "hybrid": {},
}


def predict_ensemble(train, y, test, seeds=7, profile="hybrid"):
    """Frames must already have literature/presence attached."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    if profile in {"blend", "hybrid"}:
        names = ("context_half", "kernel") if profile == "hybrid" else ("baseline", "context", "kernel")
        components = [predict_ensemble(train, y, test, seeds, name)
                      for name in names]
        return tuple(np.mean([c[i] for c in components], axis=0) for i in range(3))
    probabilities, predictions = [], []
    for seed in range(seeds):
        if profile.startswith("kernel"):
            from .kernel import KernelSelectivityModel
            model = KernelSelectivityModel(seed=seed, **PROFILES[profile])
        elif profile.startswith("cat"):
            from .boosted import BoostedSelectivityModel
            model = BoostedSelectivityModel(seed=seed, **PROFILES[profile])
        else:
            model = SelectivityModel(seed=seed, **PROFILES[profile])
        model.fit(train, y)
        probability = model.probability(test)
        probabilities.append(probability)
        predictions.append(probability * SENTINEL + (1 - probability) * model.active_mean_)
    raw = np.mean(predictions, axis=0)
    return np.mean(probabilities, axis=0), raw, apply(raw, test)
