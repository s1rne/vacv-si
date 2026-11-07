import numpy as np
from experiments.v2_analog import tanimoto
from experiments.v4_pair_delta import fit_predict as baseline
from experiments.v6_pair_offset import fit_predict, transform, inverse


def fixture():
    x = np.ones((8, 20))
    x[np.arange(8), np.arange(8)] = 0
    x[7] = 0  # Query without anchors exercises unchanged structural fallback.
    return x, np.array([0., .01, .1, 10., 100., 5000., 36066.667, 0.]), tanimoto(x, x), np.ones((8, 8), dtype=bool), np.arange(6), np.arange(6, 8)


def test_c1_parity_with_v4():
    args = fixture()
    actual, audit = fit_predict(1., *args)
    expected = baseline("ridge-log", *args)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
    assert audit["expert_rows"] == 1


def test_heldout_targets_do_not_affect_predictions():
    args = fixture()
    before, audit = fit_predict(.01, *args)
    args[1][args[-1]] = [1e10, 55555.]
    after, other = fit_predict(.01, *args)
    np.testing.assert_array_equal(before, after)
    assert audit == other


def test_inverse_preserves_zero_and_small_values():
    y = np.array([0., .001, .01, 1., 36066.667])
    for c in (.01, .1, 1., 10.):
        np.testing.assert_allclose(inverse(transform(y, c), c), y, rtol=1e-12, atol=1e-12)
