import numpy as np
from experiments.v6_pair_background import pair_features, antisymmetric_background, fit_predict


def test_reverse_preserves_background_and_reverses_difference():
    a, b = np.array([[1., 0.], [1., 1.]]), np.array([[0., 1.], [0., 0.]])
    f, r = pair_features(a, b), pair_features(b, a)
    np.testing.assert_array_equal(f[:, :2], -r[:, :2])
    np.testing.assert_array_equal(f[:, 2:], r[:, 2:])


def test_background_conditioned_predictions_are_antisymmetric():
    class Model:
        def predict(self, x):
            return x[:, 0]*x[:, 2]+x[:, 1]**2+2*x[:, 3]+7
    a, b = np.array([[1., 0.], [1., 1.]]), np.array([[0., 1.], [0., 0.]])
    np.testing.assert_array_equal(antisymmetric_background(Model(), a, b), -antisymmetric_background(Model(), b, a))


def test_outer_holdout_targets_never_enter_pair_training():
    x = np.array([[1., 0.], [1., 1.], [0., 1.], [0., 0.]])
    sim, same = np.ones((4, 4)), np.ones((4, 4), bool)
    tr, va = np.array([0, 1, 2]), np.array([3])
    a = fit_predict("leaf2", x, np.array([1., 100., 36066.667, 0.]), sim, same, tr, va)
    b = fit_predict("leaf2", x, np.array([1., 100., 36066.667, 1e12]), sim, same, tr, va)
    np.testing.assert_array_equal(a, b)
