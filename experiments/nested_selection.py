"""Target-isolated nested selection and a small, deterministic convex grid."""
from itertools import product

import numpy as np


def simplex_grid(models, units=4):
    if models < 1 or units < 1:
        raise ValueError("Positive model count and grid denominator required")
    # Descending lexicographic order makes the first model the exact-tie default.
    return np.asarray([w for w in product(range(units, -1, -1), repeat=models)
                       if sum(w) == units], dtype=float) / units


def select_weights(predictions, targets, units=4):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if (predictions.ndim != 2 or targets.shape != (len(predictions),)
            or not len(targets) or not np.isfinite(predictions).all()
            or not np.isfinite(targets).all() or (predictions < 0).any()
            or (targets < 0).any()):
        raise ValueError("Complete aligned finite nonnegative inner OOF required")
    weights = simplex_grid(predictions.shape[1], units)
    scores = np.mean((predictions @ weights.T - targets[:, None]) ** 2, axis=0)
    individual = np.mean((predictions - targets[:, None]) ** 2, axis=0)
    return {"single_index": int(np.argmin(individual)),
            "weights": weights[np.argmin(scores)].tolist(),
            "single_mse": individual.tolist(), "grid_mse": scores.tolist()}


def evaluate_outer_fold(y, train, valid, inner, predict, model_count, units=4):
    """predict receives only fit-row targets; no outer-valid targets are read.

    All indices are global. Inner splits must partition outer train once and
    contain its exact complement for fitting, with no outer-valid rows.
    """
    y, train, valid = np.asarray(y), np.asarray(train), np.asarray(valid)
    if len(set(train)) != len(train) or len(set(valid)) != len(valid) or set(train) & set(valid):
        raise ValueError("Disjoint unique outer row indices required")
    positions = {int(row): i for i, row in enumerate(train)}
    oof = np.full((len(train), model_count), np.nan)
    seen = np.zeros(len(train), dtype=int)
    for fit, query in inner:
        fit, query = np.asarray(fit), np.asarray(query)
        if (len(set(fit)) != len(fit) or len(set(query)) != len(query)
                or set(fit) & set(query) or set(fit) | set(query) != set(train)):
            raise ValueError("Inner folds must partition only outer training rows")
        local = np.asarray([positions[int(row)] for row in query])
        pred = np.asarray(predict(fit, query, y[fit]))
        if pred.shape != (len(query), model_count):
            raise ValueError("Inner predictions have wrong shape")
        oof[local] = pred
        seen[local] += 1
    if not np.all(seen == 1):
        raise ValueError("Each outer-train row needs exactly one inner OOF prediction")
    selected = select_weights(oof, y[train], units)
    predictions = np.asarray(predict(train, valid, y[train]))
    if (predictions.shape != (len(valid), model_count)
            or not np.isfinite(predictions).all() or (predictions < 0).any()):
        raise ValueError("Invalid outer predictions")
    return {"base": predictions, "single": predictions[:, selected["single_index"]],
            "blend": predictions @ np.asarray(selected["weights"]),
            "inner_oof": oof, "selection": selected}
