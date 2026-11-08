import numpy as np

from experiments.v2_analog import tanimoto
from experiments.v2_context_analog import same_context
from experiments.v4_context_kernel import fold_predictions, predict, product_kernel


def test_product_kernel_positive_semidefinite_with_missing_context():
    rng = np.random.default_rng(5)
    x = (rng.random((24, 32)) > .7).astype(float)
    ctx = np.array([["a", "b"], ["a", "b"], ["c", "d"], ["", "b"]] * 6)
    kernel = product_kernel(tanimoto(x, x), same_context(ctx, ctx))
    assert np.allclose(kernel, kernel.T)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-10


def test_validation_targets_do_not_affect_fit_or_prediction():
    kernel = np.eye(6) + .2
    y = np.array([1., 2., 4., 8., 16., 32.])
    train, valid = np.arange(4), np.arange(4, 6)
    before = fold_predictions(kernel, y, train, valid, "product-raw-0.1")
    y[valid] = [0., 36066.667]
    after = fold_predictions(kernel, y, train, valid, "product-raw-0.1")
    np.testing.assert_array_equal(before, after)


def test_centered_intercept_reproduces_constant_for_both_scales():
    for name in ("product-raw-0.1", "product-sqrt-0.1"):
        result = predict(np.eye(4), np.zeros((2, 4)), np.full(4, 17.), name)
        np.testing.assert_allclose(result, 17.)
