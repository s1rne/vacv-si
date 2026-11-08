import numpy as np
import pytest
from experiments.v6_pair_local import local_weights, fit_predict, PROFILES
from experiments.v4_pair_delta import fit_predict as baseline_predict


@pytest.mark.parametrize("power", [4, 8])
def test_reverse_pairs_have_equal_normalized_query_weights(power):
    pairs = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])
    sim = np.array([.9, .1, .3])
    w = local_weights(sim, pairs, power)
    np.testing.assert_array_equal(w[::2], w[1::2])
    np.testing.assert_allclose(w.mean(), 1.)
    expected = .1+.9*np.array([.9, .9, .3, .3])**power
    np.testing.assert_allclose(w, expected/expected.mean())


@pytest.mark.parametrize("profile", PROFILES)
def test_query_target_cannot_affect_pair_fit_or_local_weights(profile):
    x = np.array([[1., 0.], [1., 1.], [0., 1.], [0., 0.]])
    sim, same = np.ones((4, 4)), np.ones((4, 4), bool)
    tr, va = np.array([0, 1, 2]), np.array([3])
    a = fit_predict(profile, x, np.array([1., 100., 36066.667, 0.]), sim, same, tr, va)
    b = fit_predict(profile, x, np.array([1., 100., 36066.667, 1e12]), sim, same, tr, va)
    np.testing.assert_array_equal(a, b)


def test_identical_pair_features_produce_zero_delta():
    x = np.ones((3, 2))
    sim = np.ones((3, 3))
    actual = fit_predict("local4", x, np.array([1., 3., 1e9]), sim, sim.astype(bool), np.array([0, 1]), np.array([2]))
    np.testing.assert_allclose(actual, [2.])


def test_uniform_query_weights_reproduce_global_ridge_with_unused_columns():
    x = np.array([[1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])
    y = np.array([1., 100., 36066.667, 0.])
    sim, same = np.ones((4, 4)), np.ones((4, 4), bool)
    tr, va = np.array([0, 1, 2]), np.array([3])
    actual = fit_predict("local4", x, y, sim, same, tr, va)
    expected = baseline_predict("ridge-log", x, y, sim, same, tr, va)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
