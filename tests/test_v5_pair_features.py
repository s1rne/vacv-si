import numpy as np
import pandas as pd

from experiments.v5_pair_features import representations, PROFILES
from experiments.v4_pair_delta import fit_predict
from experiments.v2_analog import fingerprint, tanimoto


def test_representations_deterministic_and_structure_only():
    f = pd.DataFrame({"SMILES": ["CCO", "CCCO", "CCO"]})
    a, b = representations(f), representations(f.assign(**{"SI VACV": [999, 0, 3]}))
    for key in PROFILES:
        np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a[key][0], a[key][2])
        assert np.isfinite(a[key]).all()
    assert a[PROFILES[2]].shape == (3, 4096)
    np.testing.assert_array_equal(a[PROFILES[2]][:, :2048] * np.sqrt(2), fingerprint(f))


def test_query_labels_do_not_change_predictions():
    f = pd.DataFrame({"SMILES": ["CCOc1ccccc1", "CCOc1ccncc1", "COc1ccccc1", "CCOc1ccccc1"]})
    original = fingerprint(f)
    sim = tanimoto(original, original)
    match = np.ones((4, 4), bool)
    y = np.array([10., 20., 30., 40.])
    changed = np.array([10., 20., 30., 1e12])
    for x in representations(f).values():
        p, gate, n = fit_predict("ridge-log", x, y, sim, match, np.array([0, 1, 2]), np.array([3]))
        q, _, _ = fit_predict("ridge-log", x, changed, sim, match, np.array([0, 1, 2]), np.array([3]))
        assert gate[0] and n > 0
        np.testing.assert_array_equal(p, q)
