"""Fixed numeric regression backends using training targets exclusively.

Output order is et-raw, et-log, cat-raw, cat-log, kernel-raw, kernel-log,
followed by et-log-external when all external arguments are supplied.
BACKEND_CONFIG records the exact production parameters; factories allow small
test estimators without changing those parameters. Log backends fit log1p(y)
and predict expm1, and all predictions have only a zero floor (no upper cap).
External rows are concatenated, never pooled, and affect only the seventh model.
"""

from types import MappingProxyType

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor


_ET_PARAMS = MappingProxyType(dict(
    n_estimators=300, min_samples_leaf=2, max_features=0.5,
    random_state=900, n_jobs=2,
))
_CAT_PARAMS = MappingProxyType(dict(
    iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=5,
    loss_function="RMSE", thread_count=2, random_seed=900,
    verbose=False, allow_writing_files=False,
))
_KERNEL_PARAMS = MappingProxyType(dict(alpha=1.0, mean_prior=True))


def _config(estimator, transform, params, external=False):
    return MappingProxyType(dict(
        estimator=estimator, transform=transform, params=params, external=external,
    ))


BACKEND_CONFIG = MappingProxyType({
    "et-raw": _config("ExtraTreesRegressor", "identity", _ET_PARAMS),
    "et-log": _config("ExtraTreesRegressor", "log1p", _ET_PARAMS),
    "cat-raw": _config("CatBoostRegressor", "identity", _CAT_PARAMS),
    "cat-log": _config("CatBoostRegressor", "log1p", _CAT_PARAMS),
    "kernel-raw": _config("KernelRidge", "identity", _KERNEL_PARAMS),
    "kernel-log": _config("KernelRidge", "log1p", _KERNEL_PARAMS),
    "et-log-external": _config("ExtraTreesRegressor", "log1p", _ET_PARAMS, True),
})


def make_extra_trees():
    """Construct the fixed production Extra Trees regressor."""
    return ExtraTreesRegressor(**_ET_PARAMS)


def make_catboost():
    """Construct the fixed production CatBoost regressor, without early stopping."""
    return CatBoostRegressor(**_CAT_PARAMS)


def _numeric(value, name, ndim):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.ndim != ndim:
        raise ValueError(f"{name} must be a numeric {ndim}D array")
    array = np.array(array, dtype=float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _targets(value, name, rows):
    targets = _numeric(value, name, 1)
    if targets.shape != (rows,) or (targets < 0).any():
        raise ValueError(f"{name} must have {rows} nonnegative training targets")
    return targets


def _finish(predictions, rows, logarithmic):
    predictions = _numeric(predictions, "predictions", 1)
    if predictions.shape != (rows,):
        raise ValueError("Predictions must have one value per query row")
    if logarithmic:
        with np.errstate(over="raise", invalid="raise"):
            try:
                predictions = np.expm1(predictions)
            except FloatingPointError as error:
                raise ValueError("Inverse log predictions overflowed") from error
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must remain finite after inverse transform")
    return np.maximum(predictions, 0)


def predict_backends(train_x, train_y, query_x, train_kernel, query_kernel,
                     external_x=None, external_y=None, external_weight=None):
    """Return fixed backend predictions without any query-target argument.

    Kernel shapes are (n_train, n_train) and (n_query, n_train). All external
    arguments must be supplied together. External weights may be a nonnegative
    scalar or one value per row; training rows always weigh one. Zero-weight
    external rows are excluded so a disabled external control exactly matches
    et-log. Inputs are copied and are never mutated.
    """
    train_x = _numeric(train_x, "train_x", 2)
    query_x = _numeric(query_x, "query_x", 2)
    rows, features = train_x.shape
    query_rows = len(query_x)
    if not rows or not features or not query_rows or query_x.shape[1] != features:
        raise ValueError("Nonempty aligned training and query features required")
    train_y = _targets(train_y, "train_y", rows)
    train_kernel = _numeric(train_kernel, "train_kernel", 2)
    query_kernel = _numeric(query_kernel, "query_kernel", 2)
    if train_kernel.shape != (rows, rows) or query_kernel.shape != (query_rows, rows):
        raise ValueError("Kernel matrices must align with training and query rows")

    external_supplied = [value is not None for value in
                         (external_x, external_y, external_weight)]
    if any(external_supplied) and not all(external_supplied):
        raise ValueError("external_x, external_y, external_weight are required together")
    if all(external_supplied):
        external_x = _numeric(external_x, "external_x", 2)
        if external_x.shape[1] != features:
            raise ValueError("External features must align with training features")
        external_y = _targets(external_y, "external_y", len(external_x))
        weights = np.asarray(external_weight)
        if weights.ndim == 0:
            weights = _numeric(weights, "external_weight", 0)
            if weights < 0:
                raise ValueError("External weights must be nonnegative")
            weights = np.full(len(external_x), float(weights))
        else:
            weights = _numeric(weights, "external_weight", 1)
        if weights.shape != (len(external_x),) or (weights < 0).any():
            raise ValueError("External weights must be nonnegative and align with rows")
        enabled = weights > 0
        external_x, external_y, weights = (
            external_x[enabled], external_y[enabled], weights[enabled])

    result = {}
    for name, config in BACKEND_CONFIG.items():
        logarithmic = config["transform"] == "log1p"
        targets = np.log1p(train_y) if logarithmic else train_y
        if config["external"]:
            if not all(external_supplied):
                continue
            if not len(external_y):
                result[name] = result["et-log"].copy()
                continue
            estimator = make_extra_trees()
            estimator.fit(
                np.concatenate((train_x, external_x)),
                np.concatenate((targets, np.log1p(external_y))),
                sample_weight=np.concatenate((np.ones(rows), weights)),
            )
            predictions = estimator.predict(query_x)
        elif config["estimator"] == "KernelRidge":
            prior = targets.mean()
            coefficients = np.linalg.solve(train_kernel + np.eye(rows), targets - prior)
            predictions = query_kernel @ coefficients + prior
        else:
            estimator = (make_extra_trees() if config["estimator"] == "ExtraTreesRegressor"
                         else make_catboost())
            estimator.fit(train_x, targets)
            predictions = estimator.predict(query_x)
        result[name] = _finish(predictions, query_rows, logarithmic)
    return result
