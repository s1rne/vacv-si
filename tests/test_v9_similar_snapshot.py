import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments import v9_similar_snapshot as snapshot


@pytest.mark.parametrize("text, expected", [
    ("CHEMBL_ACT:123|CHEMBL_ACT:456", "123/456"),
    ("activity 123 (EC50); activity 456 (CC50)", "123/456"),
    ("CHEMBL999:123:456", "123/456"),
    ("EC50 activity 123 (CHEMBL1); CC50 activity 456 (CHEMBL2); molecule CHEMBL3", "123/456"),
    ("123/456", "123/456"), ("PubChem AID 123 + AID 456", None),
    ("activity 123", None),
])
def test_record_ids_are_normalized_without_confusing_assay_ids(text, expected):
    assert snapshot.record_pair(text) == expected


def test_snapshot_excludes_competition_limits_zero_and_conflicting_pairs(monkeypatch, tmp_path):
    rows = []
    def add(smiles, pair, ic=2., cc=4., relation="=", flag=False):
        rows.append(dict(smiles=smiles, source_id=pair, ic50_um=ic, cc50_um=cc,
                         ic50_relation=relation, cc50_relation="=", doi="document",
                         source="ChEMBL IC50", is_competition_compound=flag))
    add("train", "1/2")
    add("test", "3/4")
    add("external", "5/6")
    add("external", "CHEMBL_ACT:5|CHEMBL_ACT:6")
    add("external", "7/8", cc=6.)
    add("bounded", "9/10", relation=">")
    add("zero", "11/12", cc=0.)
    add("conflict", "13/14", cc=3.)
    add("conflict", "CHEMBL_ACT:13|CHEMBL_ACT:14", cc=9.)
    add("unparsed", "PubChem 15/16")
    raw = pd.DataFrame(rows).to_csv(index=False).encode()
    monkeypatch.setattr(snapshot.subprocess, "check_output", lambda command: raw)
    monkeypatch.setattr(snapshot, "load", lambda *a, **k: SimpleNamespace(
        train=pd.DataFrame({"SMILES": ["train"]}), test=pd.DataFrame({"SMILES": ["test"]})))
    monkeypatch.setattr(snapshot, "standardize", lambda s: SimpleNamespace(inchikey=s.ljust(14, "x")+"-suffix"))
    monkeypatch.setattr(snapshot, "OUT", tmp_path / "snapshot")
    snapshot.main()
    out = pd.read_csv(snapshot.OUT / "provisional-exact.csv")
    assert out.record_pair.tolist() == ["5/6", "7/8"]
    assert out.target.tolist() == [2., 3.]
    assert len(out.SMILES.unique()) == 1
    np.testing.assert_allclose(out.weight.sum(), .25)
    info = json.loads((snapshot.OUT / "manifest.json").read_text())
    assert info["competition_overlap_rows"] == info["unflagged_overlap_rows"] == 2
    exclusions = pd.read_csv(snapshot.OUT / "exclusions.csv").fillna("")
    assert exclusions.loc[7, "reasons"] == exclusions.loc[8, "reasons"] == "conflicting_record_pair"
    with pytest.raises(FileExistsError):
        snapshot.main()
