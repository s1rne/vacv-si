import zipfile
import argparse

import pandas as pd
import pytest

from scripts.import_competition_v2 import import_archive
from vacv.dataset import load
from vacv.target_version import add_legacy_target_flag, check_legacy_target


def test_official_v2_import_and_preference(tmp_path):
    archive = tmp_path / "official.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("train_v2.csv", "Title,SMILES,strain,Culture,SI VACV,CC50_uM,IC50_uM_VACV\na,C,x,y,0,secret,secret\n")
        z.writestr("test.csv", "index,Title,SMILES,strain,Culture,CC50_uM,IC50_uM_VACV\n5,b,N,x,y,secret,secret\n")
        z.writestr("sample_submit.csv", "index,SI VACV\n5,0\n")
    out = tmp_path / "v2"
    manifest = import_archive(archive, out)
    assert manifest["target_version"] == "train_v2"
    assert "secret" not in (out / "train_v2.csv").read_text()
    assert "secret" not in (out / "test.csv").read_text()
    pd.DataFrame({"SI VACV": [10000]}).to_csv(out / "train.csv", index=False)
    data = load(out, require_v2=True)
    assert data.y.tolist() == [0.]
    assert data.test["index"].tolist() == [5]
    with pytest.raises(FileExistsError):
        import_archive(archive, out)


def test_v2_does_not_silently_use_old_targets(tmp_path):
    (tmp_path / "train.csv").write_text("SI VACV\n10000\n")
    with pytest.raises(FileNotFoundError, match="train_v2"):
        load(tmp_path, require_v2=True)


def test_historical_target_requires_explicit_opt_in():
    parser = argparse.ArgumentParser()
    add_legacy_target_flag(parser)
    with pytest.raises(SystemExit) as error:
        check_legacy_target(parser, parser.parse_args([]))
    assert error.value.code == 2
    check_legacy_target(parser, parser.parse_args(["--allow-legacy-target"]))
