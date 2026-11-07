import numpy as np
from experiments.v4_pair_delta import training_pairs, antisymmetric_predict, fit_predict


def test_pair_endpoints_cannot_include_outer_holdout():
    sim = np.ones((6, 6))
    matching = np.ones_like(sim, dtype=bool)
    train = np.array([0, 2, 5])
    pairs = training_pairs(sim, matching, train)
    assert set(pairs.ravel()) == set(train)
    assert all(a != b for a, b in pairs)
    assert set(map(tuple, pairs)) == set(map(tuple, pairs[:, ::-1]))
    assert len(pairs) == 6


def test_no_context_or_low_similarity_means_no_pairs():
    assert training_pairs(np.ones((3, 3)), np.zeros((3, 3), bool), np.arange(3)).shape == (0, 2)
    assert training_pairs(np.eye(3), np.ones((3, 3), bool), np.arange(3)).shape == (0, 2)


def test_nonlinear_prediction_is_antisymmetric():
    class Nonlinear:
        def predict(self, x):
            return 7 + x[:, 0]**2 + 3*x[:, 1]
    x = np.array([[2., -1.], [5., 3.]])
    p = antisymmetric_predict(Nonlinear(), x)
    np.testing.assert_allclose(p, -antisymmetric_predict(Nonlinear(), -x))
    np.testing.assert_allclose(p, 3*x[:, 1])


def test_changing_outer_holdout_targets_cannot_change_prediction():
    x = np.array([[1., 0.], [1., 1.], [0., 1.]])
    sim = np.ones((3, 3))
    same = np.ones((3, 3), dtype=bool)
    tr, va = np.array([0, 1]), np.array([2])
    first = fit_predict("ridge-raw", x, np.array([1., 100., 0.]), sim, same, tr, va)[0]
    second = fit_predict("ridge-raw", x, np.array([1., 100., 1e12]), sim, same, tr, va)[0]
    np.testing.assert_array_equal(first, second)
