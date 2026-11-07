import numpy as np
import pytest
from experiments.v5_pair_tailweight import tail_weights, fit_predict, PROFILES


@pytest.mark.parametrize("power", [.125, .25, .5])
def test_weights_are_symmetric_exact_and_mean_one(power):
    a, b = np.array([0., 10., 36066.667]), np.array([10., 0., 100.])
    w = tail_weights(a, b, power)
    exact = ((1+a)*(1+b))**power
    np.testing.assert_allclose(w, exact/exact.mean())
    np.testing.assert_array_equal(w, tail_weights(b, a, power))
    assert w[0] == w[1]
    assert w[2] > w[0]
    assert tail_weights(np.array([]), np.array([]), power).size == 0


@pytest.mark.parametrize("profile", PROFILES)
def test_outer_holdout_target_does_not_affect_fitted_model_or_prediction(profile):
    x = np.array([[1., 0.], [1., 1.], [0., 1.], [0., 0.]])
    sim, same = np.ones((4, 4)), np.ones((4, 4), bool)
    tr, va = np.array([0, 1, 2]), np.array([3])
    a = fit_predict(profile, x, np.array([1., 100., 36066.667, 0.]), sim, same, tr, va)
    b = fit_predict(profile, x, np.array([1., 100., 36066.667, 1e12]), sim, same, tr, va)
    np.testing.assert_array_equal(a, b)
