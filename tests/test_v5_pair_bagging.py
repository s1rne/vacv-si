import numpy as np
from experiments.v2_analog import tanimoto
from experiments.v5_pair_bagging import fit_bags, molecule_pair_weights


def test_molecule_weights_are_positive_symmetric_normalized_and_repeatable():
    pairs = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])
    w = molecule_pair_weights(pairs, 3)
    assert (w > 0).all()
    np.testing.assert_allclose(w.mean(axis=1), 1.)
    np.testing.assert_array_equal(w[:, 0], w[:, 1])
    np.testing.assert_array_equal(w[:, 2], w[:, 3])
    np.testing.assert_array_equal(w, molecule_pair_weights(pairs, 3))


def test_heldout_targets_cannot_affect_bags_or_anchor_support():
    x = np.ones((8, 20))
    x[np.arange(8), np.arange(8)] = 0
    sim = tanimoto(x, x)
    match = np.ones_like(sim, dtype=bool)
    y = np.array([1., 2., 3., 8., 15., 25., 100., 36066.667])
    tr, va = np.arange(6), np.arange(6, 8)
    before, audit = fit_bags(x, y, sim, match, tr, va)
    y[va] = [0., 1e10]
    after, audit_after = fit_bags(x, y, sim, match, tr, va)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    assert audit == audit_after
    assert audit["expert_rows"] == 2
    assert all(set(a).issubset(set(tr)) for a in audit["anchors"])
