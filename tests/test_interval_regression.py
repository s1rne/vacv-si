import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from vacv.interval_regression import IntervalKernelRegressor, interval_loss_gradient


def test_interval_loss_direction_and_analytic_gradient():
    lower = np.array([2., -np.inf, 3., 4.])
    upper = np.array([np.inf, 2., 3., 6.])
    prediction = np.array([1., 5., 4., 5.])
    weights = np.array([.25, .5, 1., 1.])
    loss, gradient = interval_loss_gradient(prediction, lower, upper, weights)
    assert loss == pytest.approx(2.875)
    np.testing.assert_allclose(gradient, [-.25, 1.5, 1., 0.])
    numerical = approx_derivative(lambda p: interval_loss_gradient(p, lower, upper, weights)[0], prediction)
    np.testing.assert_allclose(gradient, numerical.ravel(), atol=1e-7)
    assert interval_loss_gradient(np.array([3., 1., 3., 5.]), lower, upper, weights)[0] == 0


def test_exact_observations_match_centered_kernel_ridge():
    x = np.array([[1., 0.], [0., 1.], [1., 1.], [.4, .7]])
    kernel = x@x.T
    y = np.array([0., 1., 3., 10.])
    alpha = 2.
    xc, yc = x-x.mean(0), y-y.mean()
    coefficient = np.linalg.solve(xc.T@xc+alpha*np.eye(2), xc.T@yc)
    expected = xc@coefficient+y.mean()
    model = IntervalKernelRegressor(alpha).fit(kernel, y, y)
    np.testing.assert_allclose(model.predict(kernel), expected, atol=1e-5)


@pytest.mark.parametrize("lower,upper", [([np.nan, 0], [1, 0]),
                                       ([2, 0], [1, 0]),
                                       ([-np.inf, 0], [np.inf, 0])])
def test_invalid_intervals_rejected(lower, upper):
    with pytest.raises(ValueError):
        IntervalKernelRegressor().fit(np.eye(2), lower, upper)


def test_feasible_bound_is_not_a_point_target():
    # Identical observations share a prediction; a lower bound below an exact
    # observation must not drag the exact observation toward its boundary.
    model = IntervalKernelRegressor().fit(np.ones((2, 2)), [5., 1.], [5., np.inf])
    np.testing.assert_allclose(model.predict(np.ones((1, 2))), [5.], atol=1e-6)
