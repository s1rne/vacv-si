"""Conditional-mean literature substitution rather than a hard point override."""

from __future__ import annotations

import numpy as np

from .pipeline import SENTINEL


def conditional_lookup(prediction, probability, frame):
    prediction = np.asarray(prediction, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if prediction.shape != (len(frame),) or probability.shape != prediction.shape:
        raise ValueError("Predictions, probabilities and frame must align")
    if (not np.isfinite(prediction).all() or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()):
        raise ValueError("Finite predictions and probabilities in [0, 1] required")
    literature = frame["literature_si"].to_numpy(dtype=float)
    found = np.isfinite(literature)
    result = prediction.copy()
    active_si = np.clip(literature[found], 0, SENTINEL)
    result[found] = probability[found] * SENTINEL + (1 - probability[found]) * active_si
    return result
