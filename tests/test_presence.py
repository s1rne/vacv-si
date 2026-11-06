import numpy as np
import pandas as pd
import pytest

from vacv.presence import _auc, _neighbour_columns


def test_auc_ranks_ties_to_half():
    labels = np.array([True, True, False, False])
    assert _auc(labels, np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.5)
    assert _auc(labels, np.array([2.0, 2.0, 1.0, 1.0])) == pytest.approx(1.0)
    assert _auc(labels, np.array([1.0, 1.0, 2.0, 2.0])) == pytest.approx(0.0)


def test_auc_treats_missing_as_lowest():
    labels = np.array([True, False])
    # The positive has no value, the negative does, so the positive ranks last.
    assert _auc(labels, np.array([np.nan, 1.0])) == pytest.approx(0.0)


def test_auc_needs_both_classes():
    assert np.isnan(_auc(np.array([True, True]), np.array([1.0, 2.0])))


def test_neighbours_exclude_the_compound_itself():
    reference = pd.DataFrame(
        {
            "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"],
            "inchikey14": ["BSYNRYMUTXBX", "HEFNNWSXXWATRW"],
            "si": [10.0, 99.0],
            "si_quality": ["measured", "measured"],
        }
    )
    split = pd.DataFrame(
        {
            "SMILES": ["CC(=O)Oc1ccccc1C(=O)O"],
            "inchikey14": ["BSYNRYMUTXBX"],
        }
    )
    neighbours = _neighbour_columns(split, reference)
    # Aspirin is in the reference set, so a naive nearest neighbour would be
    # itself at similarity 1.0 and would leak its own SI back as a feature.
    assert neighbours.loc[0, "nn1_tanimoto"] < 1.0
    assert neighbours.loc[0, "nn1_inchikey14"] == "HEFNNWSXXWATRW"
    assert neighbours.loc[0, "nn1_si"] == pytest.approx(99.0)


def test_neighbours_handle_unparsable_smiles():
    reference = pd.DataFrame(
        {
            "smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
            "inchikey14": ["BSYNRYMUTXBX"],
            "si": [10.0],
            "si_quality": ["measured"],
        }
    )
    split = pd.DataFrame({"SMILES": ["not a molecule ("], "inchikey14": [None]})
    neighbours = _neighbour_columns(split, reference)
    assert neighbours.empty or neighbours.isna().all().all()
