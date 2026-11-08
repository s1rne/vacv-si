import numpy as np
import pandas as pd
import pytest

from vacv.recovery_model import sharpen_probability
from vacv.clean_features import attach_clean


def test_probability_boundaries_and_symmetry():
    p = np.array([0., .2, .5, .8, 1.])
    q = sharpen_probability(p)
    assert np.isfinite(q).all()
    assert np.all(np.diff(q) > 0)
    assert q[0] == 0 and q[-1] == 1 and q[2] == .5
    np.testing.assert_allclose(q, 1 - sharpen_probability(1 - p))


@pytest.mark.parametrize("p,exponent", [([np.nan], 1.2), ([-.1], 1.2), ([1.1], 1.2),
                                       ([.5], 0), ([.5], np.inf)])
def test_invalid_probability_is_rejected(p, exponent):
    with pytest.raises(ValueError):
        sharpen_probability(p, exponent)


def test_clean_features_do_not_depend_on_targets_or_forbidden_columns(monkeypatch):
    monkeypatch.setattr("vacv.clean_features.presence", lambda: pd.DataFrame({"SMILES": ["C"]}))
    monkeypatch.setattr("vacv.clean_features.PRESENCE_FEATURES", [])
    raw = pd.DataFrame({"index": [4], "SMILES": ["C"], "Title": ["1"],
                        "strain": ["x"], "Culture": ["y"]})
    poisoned = raw.assign(**{"SI VACV": 999., "CC50_uM": 123., "IC50_uM_VACV": 456.,
                             "recommended_si": 789., "matched": 1.})
    pd.testing.assert_frame_equal(attach_clean(raw), attach_clean(poisoned))
