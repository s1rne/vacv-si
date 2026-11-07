"""Synthetic regressions for target isolation and nested model selection."""

import numpy as np
import pytest

from experiments.nested_selection import (
    evaluate_outer_fold,
    select_weights,
    simplex_grid,
)


def _problem():
    targets = np.arange(1, 9, dtype=float) * 10
    train = np.array([4, 0, 5, 2])
    valid = np.array([1, 3, 6, 7])
    inner = [(train[2:], train[:2]), (train[:2], train[2:])]
    return targets, train, valid, inner


def _predict(fit, query, fit_targets):
    features = np.asarray(query, dtype=float) / 10
    return np.column_stack((fit_targets.mean() + features,
                            fit_targets.mean() * 0.8 + features * 2))


def test_outer_targets_are_never_passed_to_predictor_or_used_for_selection():
    targets, train, valid, inner = _problem()
    calls = []

    def predictor(fit, query, fit_targets):
        np.testing.assert_array_equal(fit_targets, targets[fit])
        assert not set(fit) & set(valid)
        assert not set(fit) & set(query)
        assert not set(fit_targets) & set(targets[valid])
        if set(query) != set(valid):
            assert not set(fit_targets) & set(targets[query])
        calls.append((fit.copy(), query.copy(), fit_targets.copy()))
        return _predict(fit, query, fit_targets)

    baseline = evaluate_outer_fold(targets, train, valid, inner, predictor, 2)
    changed = targets.copy()
    changed[valid] = [1e100, np.nan, np.inf, -np.inf]
    perturbed = evaluate_outer_fold(changed, train, valid, inner, predictor, 2)

    assert len(calls) == 6
    for offset in (0, 3):
        for actual, expected in zip(calls[offset:offset + 3],
                                    [*inner, (train, valid)]):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
    for key in ("base", "single", "blend", "inner_oof"):
        np.testing.assert_array_equal(baseline[key], perturbed[key])
    assert baseline["selection"] == perturbed["selection"]


def test_inner_oof_and_selected_outputs_preserve_requested_row_order():
    targets, train, valid, inner = _problem()
    result = evaluate_outer_fold(targets, train, valid, inner, _predict, 2)
    expected_oof = np.vstack([
        _predict(inner[0][0], inner[0][1], targets[inner[0][0]]),
        _predict(inner[1][0], inner[1][1], targets[inner[1][0]]),
    ])
    np.testing.assert_array_equal(result["inner_oof"], expected_oof)
    np.testing.assert_array_equal(result["base"], _predict(train, valid, targets[train]))
    np.testing.assert_array_equal(
        result["single"], result["base"][:, result["selection"]["single_index"]])
    np.testing.assert_allclose(
        result["blend"], result["base"] @ result["selection"]["weights"])


@pytest.mark.parametrize("train,valid", [
    ([0, 0, 2, 3], [4, 5]),
    ([0, 1, 2, 3], [4, 4]),
    ([0, 1, 2, 3], [3, 4]),
])
def test_duplicate_or_overlapping_outer_rows_are_rejected(train, valid):
    def must_not_predict(*args):
        pytest.fail("Malformed outer split reached predictor")

    with pytest.raises(ValueError, match="Disjoint unique"):
        evaluate_outer_fold(np.arange(1, 7), train, valid, [], must_not_predict, 2)


@pytest.mark.parametrize("fit,query", [
    ([2, 2, 3], [0, 1]),  # duplicate fitting row
    ([2, 3], [0, 0, 1]),  # duplicate query row
    ([1, 2, 3], [0, 1]),  # overlap
    ([2], [0, 1]),        # missing complement row
    ([2, 3, 4], [0, 1]),  # outer held-out row in fitting set
    ([2, 3], [0, 1, 4]),  # outer held-out row in query set
])
def test_malformed_inner_partition_is_rejected_before_prediction(fit, query):
    def must_not_predict(*args):
        pytest.fail("Malformed inner partition reached predictor")

    with pytest.raises(ValueError, match="partition only outer training"):
        evaluate_outer_fold(np.arange(1, 7), [0, 1, 2, 3], [4, 5],
                            [(fit, query)], must_not_predict, 2)


@pytest.mark.parametrize("inner", [
    [],
    [([2, 3], [0, 1])],
    [([2, 3], [0, 1]), ([2, 3], [0, 1])],
    [([2, 3], [0, 1]), ([0, 1], [2, 3]), ([2, 3], [0, 1])],
])
def test_uncovered_or_repeated_inner_query_rows_are_rejected(inner):
    calls = []

    def predictor(fit, query, fit_targets):
        assert not set(query) & {4, 5}
        calls.append(query)
        return _predict(fit, query, fit_targets)

    with pytest.raises(ValueError, match="exactly one inner OOF"):
        evaluate_outer_fold(np.arange(1, 7), [0, 1, 2, 3], [4, 5],
                            inner, predictor, 2)
    assert len(calls) == len(inner)


def test_four_model_simplex_is_complete_unique_and_baseline_first():
    weights = simplex_grid(4, units=4)
    assert weights.shape == (35, 4)
    assert np.unique(weights, axis=0).shape == weights.shape
    assert np.all(weights >= 0)
    np.testing.assert_array_equal(weights.sum(axis=1), np.ones(35))
    np.testing.assert_array_equal(weights * 4, np.round(weights * 4))
    np.testing.assert_array_equal(weights[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(weights, simplex_grid(4, units=4))
    for vertex in np.eye(4):
        assert np.any(np.all(weights == vertex, axis=1))


def test_exact_selection_ties_choose_first_baseline_deterministically():
    predictions = np.tile(np.array([1.0, 2.0, 3.0])[:, None], (1, 4))
    targets = np.array([2.0, 3.0, 4.0])
    result = select_weights(predictions, targets)
    assert result["single_index"] == 0
    assert result["weights"] == [1, 0, 0, 0]
    assert result["grid_mse"] == [1.0] * 35
    assert result == select_weights(predictions, targets)


def test_known_convex_blend_beats_every_individual_model():
    predictions = np.array([[0.0, 4.0], [4.0, 0.0], [0.0, 8.0]])
    targets = predictions @ np.array([0.25, 0.75])
    result = select_weights(predictions, targets, units=4)
    assert result["weights"] == [0.25, 0.75]
    assert result["single_index"] == 1
    assert min(result["grid_mse"]) == 0
    assert all(mse > 0 for mse in result["single_mse"])


@pytest.mark.parametrize("predictions,targets", [
    (np.ones(3), np.ones(3)),
    (np.ones((3, 2, 1)), np.ones(3)),
    (np.ones((3, 2)), np.ones((3, 1))),
    (np.ones((3, 2)), np.ones(2)),
    (np.ones((0, 2)), np.ones(0)),
    (np.ones((3, 0)), np.ones(3)),
    ([[np.nan, 1]], [1]),
    ([[np.inf, 1]], [1]),
    ([[-np.inf, 1]], [1]),
    ([[-1, 1]], [1]),
    ([[1, 1]], [np.nan]),
    ([[1, 1]], [np.inf]),
    ([[1, 1]], [-1]),
])
def test_selection_rejects_invalid_shapes_or_values(predictions, targets):
    with pytest.raises(ValueError):
        select_weights(predictions, targets)


@pytest.mark.parametrize("models,units", [(0, 4), (-1, 4), (4, 0), (4, -1)])
def test_simplex_rejects_nonpositive_dimensions(models, units):
    with pytest.raises(ValueError, match="Positive"):
        simplex_grid(models, units)


@pytest.mark.parametrize("stage", ["inner", "outer"])
@pytest.mark.parametrize("bad", ["shape", "nan", "inf", "negative"])
def test_nested_evaluation_rejects_invalid_predictor_output(stage, bad):
    targets, train, valid, inner = _problem()

    def predictor(fit, query, fit_targets):
        values = _predict(fit, query, fit_targets)
        is_outer = set(query) == set(valid)
        if is_outer == (stage == "outer"):
            if bad == "shape":
                return values[:, 0]
            values[0, 0] = {"nan": np.nan, "inf": np.inf, "negative": -1}[bad]
        return values

    with pytest.raises(ValueError):
        evaluate_outer_fold(targets, train, valid, inner, predictor, 2)
