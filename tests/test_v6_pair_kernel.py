import numpy as np
from experiments.v2_analog import tanimoto
from experiments.v6_pair_kernel import pair_gram, coefficients, fit_predict


def test_pair_kernel_psd_and_antisymmetry():
    rng = np.random.default_rng(6)
    x = (rng.random((10, 20)) > .5).astype(float)
    pairs = np.array([[0, 1], [1, 2], [2, 4], [4, 5], [1, 5]])
    for power in (1, 2):
        kernel = tanimoto(x, x) ** power
        gram = pair_gram(kernel, pairs, pairs)
        assert np.linalg.eigvalsh(gram).min() > -1e-10
        np.testing.assert_allclose(pair_gram(kernel, pairs[:, ::-1], pairs), -gram, atol=1e-14)


def test_unique_pairs_equal_both_orientations_with_adjusted_alpha():
    x = np.random.default_rng(6).normal(size=(8, 5))
    kernel = x @ x.T
    pairs = np.array([[0, 1], [1, 2], [3, 4]])
    target = np.array([1., -2., 4.])
    directed = np.concatenate([pairs, pairs[:, ::-1]])
    doubled_target = np.concatenate([target, -target])
    gram = pair_gram(kernel, directed, directed)
    coef = np.linalg.solve(gram + 10 * np.eye(6), doubled_target)
    query = np.array([[5, 6], [6, 7]])
    expected = pair_gram(kernel, query, directed) @ coef
    actual = pair_gram(kernel, query, pairs) @ coefficients(kernel, pairs, target, 10)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_heldout_target_independence():
    x = np.ones((8, 20))
    x[np.arange(8), np.arange(8)] = 0
    sim, match = tanimoto(x, x), np.ones((8, 8), dtype=bool)
    y = np.arange(8, dtype=float)
    tr, va = np.arange(6), np.arange(6, 8)
    before, audit = fit_predict("pair-kernel-p2-a1", y, sim, match, tr, va)
    y[va] = [36066.667, 1e10]
    after, other = fit_predict("pair-kernel-p2-a1", y, sim, match, tr, va)
    np.testing.assert_array_equal(before, after)
    assert audit == other and audit["expert_rows"] == 2
    assert all(set(a).issubset(set(tr)) for a in audit["anchors"])
