"""Small numeric-only fixtures for the fixed backend comparison contract."""

import inspect

import numpy as np
import pytest
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

from experiments import v9_model_backends as backends


NAMES = ("et-raw", "et-log", "cat-raw", "cat-log", "kernel-raw", "kernel-log")


@pytest.fixture
def problem():
    train_x = np.array([[0., 1.], [1., 2.], [2., 0.], [3., 1.], [4., 2.], [5., 0.]])
    query_x = np.array([[0.5, 1.5], [3.5, 0.5]])
    return dict(train_x=train_x, train_y=np.array([0., 1., 2., 6., 10., 20.]),
                query_x=query_x, train_kernel=train_x @ train_x.T,
                query_kernel=query_x @ train_x.T)


@pytest.fixture
def tiny_models(monkeypatch):
    def extra_trees():
        params = dict(backends.BACKEND_CONFIG["et-raw"]["params"])
        params.update(n_estimators=5, n_jobs=1)
        return ExtraTreesRegressor(**params)

    def catboost():
        params = dict(backends.BACKEND_CONFIG["cat-raw"]["params"])
        params.update(iterations=5, thread_count=1)
        return CatBoostRegressor(**params)

    monkeypatch.setattr(backends, "make_extra_trees", extra_trees)
    monkeypatch.setattr(backends, "make_catboost", catboost)


def test_configuration_records_exact_order_and_immutable_fixed_hyperparameters():
    assert tuple(backends.BACKEND_CONFIG) == (*NAMES, "et-log-external")
    et_params = dict(n_estimators=300, min_samples_leaf=2, max_features=0.5,
                     random_state=900, n_jobs=2)
    cat_params = dict(iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=5,
                      loss_function="RMSE", thread_count=2, random_seed=900,
                      verbose=False, allow_writing_files=False)
    assert dict(backends.BACKEND_CONFIG["et-raw"]["params"]) == et_params
    assert dict(backends.BACKEND_CONFIG["cat-log"]["params"]) == cat_params
    assert dict(backends.BACKEND_CONFIG["kernel-raw"]["params"]) == {
        "alpha": 1.0, "mean_prior": True}
    assert backends.BACKEND_CONFIG["et-log-external"]["params"] == et_params
    assert backends.make_catboost().get_params() == cat_params
    tree = backends.make_extra_trees()
    for key, expected in et_params.items():
        assert tree.get_params()[key] == expected
    with pytest.raises(TypeError):
        backends.BACKEND_CONFIG["et-raw"] = {}
    with pytest.raises(TypeError):
        backends.BACKEND_CONFIG["et-raw"]["transform"] = "log1p"
    with pytest.raises(TypeError):
        backends.BACKEND_CONFIG["et-raw"]["params"]["n_estimators"] = 1


def test_real_tiny_estimators_are_deterministic_and_preserve_inputs(problem, tiny_models):
    original = {name: value.copy() for name, value in problem.items()}
    first = backends.predict_backends(**problem)
    second = backends.predict_backends(**problem)
    assert tuple(first) == NAMES
    for name in first:
        assert first[name].shape == (2,)
        assert np.isfinite(first[name]).all()
        assert (first[name] >= 0).all()
        np.testing.assert_array_equal(first[name], second[name])
    for name in problem:
        np.testing.assert_array_equal(problem[name], original[name])


@pytest.mark.parametrize("external_weight", [0., np.zeros(3)])
def test_disabled_external_exactly_matches_baseline(problem, tiny_models, external_weight):
    control = backends.predict_backends(**problem)
    repeated = backends.predict_backends(
        **problem, external_x=np.array([[100., 200.]] * 3),
        external_y=np.array([0., 1e9, 1e100]), external_weight=external_weight)
    assert tuple(repeated) == (*NAMES, "et-log-external")
    for name in NAMES:
        np.testing.assert_array_equal(repeated[name], control[name])
    np.testing.assert_array_equal(repeated["et-log-external"], control["et-log"])
    assert not np.shares_memory(repeated["et-log-external"], repeated["et-log"])


def test_empty_external_control_matches_baseline(problem, tiny_models):
    result = backends.predict_backends(
        **problem, external_x=np.empty((0, 2)), external_y=np.empty(0),
        external_weight=np.empty(0))
    np.testing.assert_array_equal(result["et-log-external"], result["et-log"])


def test_enabled_external_never_changes_base_models(problem, tiny_models):
    base = backends.predict_backends(**problem)
    extra = backends.predict_backends(
        **problem, external_x=problem["train_x"],
        external_y=problem["train_y"] + 100, external_weight=2.)
    for name in NAMES:
        np.testing.assert_array_equal(extra[name], base[name])
    assert not np.array_equal(extra["et-log-external"], extra["et-log"])


@pytest.mark.parametrize("logarithmic", [False, True])
def test_kernel_matches_centered_alpha_one_solution(problem, tiny_models, logarithmic):
    result = backends.predict_backends(**problem)
    target = np.log1p(problem["train_y"]) if logarithmic else problem["train_y"]
    coefficients = np.linalg.solve(problem["train_kernel"] + np.eye(len(target)),
                                   target - target.mean())
    prediction = problem["query_kernel"] @ coefficients + target.mean()
    if logarithmic:
        prediction = np.expm1(prediction)
    name = "kernel-log" if logarithmic else "kernel-raw"
    np.testing.assert_allclose(result[name], np.maximum(prediction, 0))


def test_fit_calls_receive_only_training_targets_and_external_is_not_pooled(problem, monkeypatch):
    calls = []

    class RecordingEstimator:
        def fit(self, x, y, **kwargs):
            calls.append((x.copy(), y.copy(), kwargs))
            return self

        def predict(self, x):
            np.testing.assert_array_equal(x, problem["query_x"])
            return np.zeros(len(x))

    monkeypatch.setattr(backends, "make_extra_trees", RecordingEstimator)
    monkeypatch.setattr(backends, "make_catboost", RecordingEstimator)
    external_x = np.array([[0., 1.], [0., 1.], [0., 1.]])
    external_y = np.array([40., 90., 999.])
    external_weight = np.array([0.2, 0.8, 0.])
    backends.predict_backends(**problem, external_x=external_x, external_y=external_y,
                              external_weight=external_weight)
    assert tuple(inspect.signature(backends.predict_backends).parameters) == (
        "train_x", "train_y", "query_x", "train_kernel", "query_kernel",
        "external_x", "external_y", "external_weight")
    assert len(calls) == 5
    for index, (x, y, kwargs) in enumerate(calls[:4]):
        np.testing.assert_array_equal(x, problem["train_x"])
        expected_y = np.log1p(problem["train_y"]) if index % 2 else problem["train_y"]
        np.testing.assert_array_equal(y, expected_y)
        assert kwargs == {}  # no validation targets, early stopping, or evaluation sets
    x, y, kwargs = calls[-1]
    np.testing.assert_array_equal(x, np.vstack((problem["train_x"], external_x[:2])))
    np.testing.assert_array_equal(y, np.log1p(np.r_[problem["train_y"], external_y[:2]]))
    np.testing.assert_array_equal(kwargs["sample_weight"], np.r_[np.ones(6), [0.2, 0.8]])
    np.testing.assert_array_equal(external_y, [40., 90., 999.])
    np.testing.assert_array_equal(external_weight, [0.2, 0.8, 0.])


@pytest.mark.parametrize("name,value", [
    ("train_x", np.ones((0, 2))),
    ("train_x", np.ones((6, 0))),
    ("train_x", np.ones(6)),
    ("train_x", np.full((6, 2), np.nan)),
    ("train_x", np.full((6, 2), "1")),
    ("train_x", np.ones((6, 2), dtype=complex)),
    ("query_x", np.ones((0, 2))),
    ("query_x", np.ones((2, 3))),
    ("query_x", np.full((2, 2), np.inf)),
    ("train_y", np.ones(5)),
    ("train_y", np.ones((6, 1))),
    ("train_y", np.full(6, -1.)),
    ("train_y", np.full(6, np.nan)),
    ("train_y", np.full(6, np.inf)),
    ("train_kernel", np.ones((6, 5))),
    ("train_kernel", np.full((6, 6), np.inf)),
    ("query_kernel", np.ones((6, 2))),
    ("query_kernel", np.full((2, 6), np.nan)),
])
def test_invalid_base_input_fails_before_model_construction(problem, monkeypatch, name, value):
    def forbidden_factory():
        pytest.fail("Invalid inputs reached model construction")

    monkeypatch.setattr(backends, "make_extra_trees", forbidden_factory)
    problem[name] = value
    with pytest.raises(ValueError):
        backends.predict_backends(**problem)


@pytest.mark.parametrize("changes", [
    {"external_x": None},
    {"external_y": None},
    {"external_weight": None},
    {"external_x": np.ones((3, 1))},
    {"external_x": np.full((3, 2), np.inf)},
    {"external_y": np.ones(2)},
    {"external_y": np.full(3, -1.)},
    {"external_y": np.full(3, np.nan)},
    {"external_weight": np.ones(2)},
    {"external_weight": np.ones((3, 1))},
    {"external_weight": -1.},
    {"external_weight": np.array([1., -1., 1.])},
    {"external_weight": np.inf},
    {"external_weight": np.array([1., np.nan, 1.])},
])
def test_invalid_external_input_fails_before_fitting(problem, monkeypatch, changes):
    def forbidden_factory():
        pytest.fail("Invalid external inputs reached model construction")

    monkeypatch.setattr(backends, "make_extra_trees", forbidden_factory)
    external = dict(external_x=np.ones((3, 2)), external_y=np.ones(3), external_weight=1.)
    external.update(changes)
    with pytest.raises(ValueError):
        backends.predict_backends(**problem, **external)


@pytest.mark.parametrize("invalid", [np.full(2, np.nan), np.full(2, np.inf),
                                     np.ones((2, 1)), np.ones(3), np.full(2, 1000.)])
def test_invalid_or_overflowed_predictions_fail(problem, monkeypatch, invalid):
    class BadEstimator:
        def fit(self, *args, **kwargs):
            return self

        def predict(self, x):
            return invalid

    monkeypatch.setattr(backends, "make_extra_trees", BadEstimator)
    with pytest.raises(ValueError):
        backends.predict_backends(**problem)


def test_predictions_have_zero_floor_but_no_training_target_or_upper_cap(problem, monkeypatch):
    class FixedEstimator:
        def fit(self, *args, **kwargs):
            return self

        def predict(self, x):
            return np.array([-2., 30.])

    monkeypatch.setattr(backends, "make_extra_trees", FixedEstimator)
    monkeypatch.setattr(backends, "make_catboost", FixedEstimator)
    result = backends.predict_backends(**problem)
    np.testing.assert_array_equal(result["et-raw"], [0., 30.])
    np.testing.assert_array_equal(result["cat-raw"], [0., 30.])
    np.testing.assert_array_equal(result["et-log"], [0., np.expm1(30.)])
    np.testing.assert_array_equal(result["cat-log"], [0., np.expm1(30.)])
