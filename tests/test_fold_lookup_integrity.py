from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vacv import competition, coverage, literature, recover
from vacv import match_competition


def test_fold_lookup_is_invariant_to_heldout_target_changes(monkeypatch, tmp_path):
    train = pd.DataFrame({
        "split": ["train"] * 3,
        "index": [0, 1, 2],
        "Title": ["a", "b", "heldout"],
        "SMILES": ["C", "CC", "CCC"],
        "inchikey": ["k1", "k2", "k3"],
        "inchikey14": ["k1", "k2", "k3"],
        "Culture": ["Vero"] * 3,
        "strain": ["VACV"] * 3,
        "SI VACV": [2.0, 3.0, 4.0],
    })
    test = pd.DataFrame({
        "split": ["test"], "index": [0], "Title": ["test"], "SMILES": ["CCCC"],
        "inchikey": ["k4"], "inchikey14": ["k4"], "Culture": ["Vero"], "strain": ["VACV"],
    })
    current = {"train": train.copy()}
    monkeypatch.setattr(competition, "load_split", lambda name: current[name] if name == "train" else test)
    monkeypatch.setattr(literature, "CURATED", tmp_path / "curated.csv.gz")
    literature.CURATED.touch()
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(competition, "load_split", lambda name: current["train"] if name == "train" else test)

    seen = []

    class Records:
        pass

    monkeypatch.setattr(recover, "Records", lambda *args: Records())
    monkeypatch.setattr(recover, "inactive_records", lambda: pd.DataFrame())
    monkeypatch.setattr(recover, "molecule_links", lambda: pd.DataFrame())
    monkeypatch.setattr(match_competition, "open_source_rows", lambda frame: frame)

    def prior(records, split="train", frame=None):
        seen.append(frame["SI VACV"].tolist())
        return {"fold_target_sum": float(frame["SI VACV"].sum())}

    monkeypatch.setattr(coverage, "document_prior", prior)

    def fake_recover(split, records, links, prior):
        return pd.DataFrame({
            "split": split["split"], "index": split["index"], "Title": split["Title"],
            "smiles": split["SMILES"], "si": float(prior["fold_target_sum"]),
            "si_quality": "measured", "match_tier": "inchikey", "rule": "document",
            "document_id": "doc", "doi": "doi", "cc50": 2.0, "ic50": 1.0,
        })

    monkeypatch.setattr(recover, "recover", fake_recover)
    first = literature.fold_table(np.array([0, 1]))
    current["train"].loc[2, "SI VACV"] = 999999.0
    second = literature.fold_table(np.array([0, 1]))

    assert seen == [[2.0, 3.0], [2.0, 3.0]]
    pd.testing.assert_frame_equal(first, second)


def test_fold_lookup_requires_curated_corpus(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the repository curation pipeline"):
        literature._rebuild_fold_lookup(np.array([0]), tmp_path / "absent.csv.gz")


def test_competition_loader_does_not_read_forbidden_columns(monkeypatch, tmp_path):
    allowed = {"SMILES", "Title", "strain", "Culture", "SI VACV"}
    real_read_csv = pd.read_csv
    seen = {}

    def read_csv(*args, **kwargs):
        seen["usecols"] = kwargs["usecols"]
        return real_read_csv(*args, **kwargs)

    path = tmp_path / "train.csv"
    path.write_text("SMILES,Title,strain,Culture,SI VACV,CC50_uM,IC50_uM_VACV\nC,x,VACV,Vero,2,1,0.5\n")
    monkeypatch.setattr(competition, "KAGGLE", tmp_path)
    monkeypatch.setattr(pd, "read_csv", read_csv)
    monkeypatch.setattr(competition, "standardize", lambda value, **kwargs: SimpleNamespace(
        inchikey="ABCDEF01234567-ABCDEFGHIJ-A", smiles="C", reject=None
    ))
    frame = competition.load_split("train")
    assert "CC50_uM" not in frame and "IC50_uM_VACV" not in frame
    assert all(seen["usecols"](name) == (name in allowed) for name in (
        "CC50_uM", "IC50_uM_VACV", "SI VACV", "SMILES", "Title"
    ))
