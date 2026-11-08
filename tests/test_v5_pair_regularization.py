import numpy as np
import pytest
from experiments.v5_pair_regularization import degree_weights, fit_predict, PROFILES


def test_degree_weights_preserve_both_orientations():
    pairs = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [2, 3], [3, 2]])
    weights = degree_weights(pairs)
    np.testing.assert_allclose(weights.mean(), 1.)
    np.testing.assert_array_equal(weights[::2], weights[1::2])
    assert weights[0] > weights[2]
    assert degree_weights(np.empty((0, 2), dtype=int)).size == 0


@pytest.mark.parametrize("profile", PROFILES)
def test_validation_targets_do_not_change_predictions(profile):
    x = np.array([[1., 0.], [1., 1.], [0., 1.]])
    sim, same = np.ones((3, 3)), np.ones((3, 3), bool)
    tr, va = np.array([0, 1]), np.array([2])
    a = fit_predict(profile, x, np.array([1., 100., 0.]), sim, same, tr, va)
    b = fit_predict(profile, x, np.array([1., 100., 1e12]), sim, same, tr, va)
    np.testing.assert_array_equal(a, b)
