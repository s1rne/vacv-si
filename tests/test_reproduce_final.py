import numpy as np
import pandas as pd
import pytest

from scripts.reproduce_final import reproduce, validate_submission


def sample():
    return pd.DataFrame({"index": [10, 20, 30], "SI VACV": [0., 1., 2.]})


@pytest.mark.parametrize("values", [[0., -1., 2.], [0., np.nan, 2.], [0., np.inf, 2.]])
def test_rejects_invalid_predictions(values):
    frame = sample()
    frame["SI VACV"] = values
    with pytest.raises(ValueError):
        validate_submission(frame, sample(), expected_rows=3)


def test_rejects_reordered_and_duplicate_indices():
    for indices in ([20, 10, 30], [10, 10, 30]):
        frame = sample()
        frame["index"] = indices
        with pytest.raises(ValueError):
            validate_submission(frame, sample(), expected_rows=3)


def test_rejects_malformed_schema_and_row_count():
    for frame in (sample().rename(columns={"SI VACV": "prediction"}), sample().iloc[:2]):
        with pytest.raises(ValueError):
            validate_submission(frame, sample(), expected_rows=3)


@pytest.mark.parametrize("suffix", [".csv", ".verification.json"])
def test_refuses_overwrite_before_loading_data(tmp_path, suffix):
    out = tmp_path / "submission.csv"
    occupied = out.with_suffix(suffix)
    occupied.write_text("preserve me")
    with pytest.raises(FileExistsError):
        reproduce(tmp_path / "missing-data", out)
    assert occupied.read_text() == "preserve me"
