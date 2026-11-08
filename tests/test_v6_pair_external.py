import hashlib

import numpy as np
import pandas as pd
import pytest

from experiments.v6_pair_external import load_external, fit_predict, source_context
from experiments.v2_analog import fingerprint, tanimoto
from experiments.v2_context_analog import same_context


def test_conservative_context_aliases_and_ambiguous_strains():
    frame = pd.DataFrame({"strain": ["Vaccinia Western Reserve", "WR", "Copenhagen and WR", "Copenhagen, cowpox Grishak", "Vaccinia virus", "Copenhagen"],
                          "Culture": ["BSC-1", "BSC1", "Vero", "Vero", "Vero", "Human foreskin fibroblast"]})
    assert source_context(frame).tolist() == [["wr", "bsc-1"], ["wr", "bsc-1"], ["", "vero"], ["", "vero"], ["", "vero"], ["copenhagen", "hff"]]


def test_snapshot_hash_and_overlap_exclusion(tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame({"SMILES": ["CCOc1ccccc1", "c1ccccc1OCC", "CCNc1ccccc1", "CCCOc1ccccc1"], "kind": ["exact"] * 4,
                  "lower_si": [1., 2., 3., 4.], "upper_si": [1., 2., 3., 4.],
                  "strain": ["WR"] * 4, "cell": ["BSC-1"] * 4}).to_csv(path, index=False)
    tr, te = pd.DataFrame({"SMILES": ["CCOc1ccccc1"]}), pd.DataFrame({"SMILES": ["CCNc1ccccc1"]})
    with pytest.raises(ValueError, match="SHA"):
        load_external(tr, te, path, "wrong")
    clean, excluded = load_external(tr, te, path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert clean.SMILES.tolist() == ["CCCOc1ccccc1"] and len(excluded) == 3


def test_validation_labels_never_fit_and_external_not_anchors():
    frame = pd.DataFrame({"SMILES": ["CCOc1ccccc1", "COc1ccccc1", "CCOc1ccccc1", "CCOc1ccncc1", "COc1ccncc1"],
                          "strain": ["WR"] * 5, "Culture": ["BSC-1"] * 5})
    x = fingerprint(frame)
    sim = tanimoto(x, x)
    matching = same_context(source_context(frame), source_context(frame))
    y = np.array([10., 20., 30., 40., 10000.])
    changed = y.copy()
    changed[2] = 1e12
    for profile in ("external10", "cross10", "cross50"):
        p, _, audit = fit_predict(profile, x, y, sim, matching, matching, np.array([0, 1]), np.array([2]), 3)
        q, _, _ = fit_predict(profile, x, changed, sim, matching, matching, np.array([0, 1]), np.array([2]), 3)
        np.testing.assert_array_equal(p, q)
        assert 2 not in audit["pair_rows"] and audit["external_pairs"] > 0
        # No competition context match: external analogs cannot act as anchors.
        no_query_match = matching.copy()
        no_query_match[2, :3] = False
        _, fired, _ = fit_predict(profile, x, y, sim, no_query_match, matching, np.array([0, 1]), np.array([2]), 3)
        assert not fired[0]
