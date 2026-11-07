import numpy as np
import pandas as pd

from vacv import clean_features
from vacv.dataset import load
from vacv.literature import PRESENCE_FEATURES


def test_clean_features_preserve_rows_and_exclude_targets(monkeypatch):
    reference = pd.DataFrame({"SMILES": ["C", "N"],
                              **{c: [1., 2.] for c in PRESENCE_FEATURES}})
    monkeypatch.setattr(clean_features, "presence", lambda: reference)
    frame = pd.DataFrame({"index": [9, 3, 2], "SMILES": ["N", "C", "N"],
                          "SI VACV": [10000., 1., 2.], "CC50_uM": [1., 2., 3.],
                          "IC50_uM_VACV": [4., 5., 6.]})
    result = clean_features.attach_clean(frame)
    assert result["index"].tolist() == [9, 3, 2]
    assert not {"SI VACV", "CC50_uM", "IC50_uM_VACV"} & set(result)
    assert result["literature_si"].isna().all()
    assert (result["matched"] == 0).all()
    frame["SI VACV"] = 123456.
    pd.testing.assert_frame_equal(result, clean_features.attach_clean(frame))


def test_loader_allows_only_train_target(tmp_path):
    frame = pd.DataFrame({"index": [0], "SMILES": ["C"], "Title": ["a"],
                          "strain": ["x"], "Culture": ["y"], "SI VACV": [42.],
                          "CC50_uM": [np.nan], "IC50_uM_VACV": [1.],
                          "unexpected_proxy": [123.]})
    for split in ("train", "test"):
        frame.to_csv(tmp_path / f"{split}.csv", index=False)
    data = load(tmp_path)
    assert data.y.tolist() == [42.]
    assert set(data.test) == {"index", "SMILES", "Title", "strain", "Culture"}
    assert set(data.train) == set(data.test) | {"SI VACV"}
