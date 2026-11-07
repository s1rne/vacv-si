import numpy as np
import pandas as pd

from experiments.v5_pair_anchor_gate import fit_predict
from experiments.v4_pair_delta import fit_predict as v4_predict
from experiments.v2_analog import fingerprint, tanimoto


def setup_data():
    frame = pd.DataFrame({"SMILES": ["CCOc1ccccc1", "CCOc1ccncc1", "COc1ccccc1", "CCOc1ccccc1"]})
    x = fingerprint(frame)
    return x, np.array([10., 20., 30., 40.]), tanimoto(x, x), np.ones((4, 4), bool), np.array([0, 1, 2]), np.array([3])


def test_v4_exact_equivalence_at_original_threshold():
    args = setup_data()
    actual = fit_predict(.7, *args)
    expected = v4_predict("ridge-log", *args)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_holdout_label_independence_and_relaxed_gate():
    x, y, sim, match, tr, va = setup_data()
    sim[3, :3] = .60
    changed = y.copy()
    changed[va] = 1e12
    for threshold in (.55, .60, .65):
        p, fired, n = fit_predict(threshold, x, y, sim, match, tr, va)
        q, _, _ = fit_predict(threshold, x, changed, sim, match, tr, va)
        np.testing.assert_array_equal(p, q)
        assert n > 0 and fired[0] == (threshold <= .60)
