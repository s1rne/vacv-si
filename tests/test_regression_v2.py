import numpy as np
import pandas as pd
import pytest

from vacv.regression_v2 import RegressionV2, allowed


def frame():
    return pd.DataFrame({"SMILES": ["C", "CC", "CCC", "N", "CN", "CO"],
                         "Title": ["a", "b", "c", "d", "e", "f"],
                         "strain": ["x"]*6, "Culture": ["Vero"]*6,
                         "SI VACV": [99.]*6, "CC50_uM": [100.]*6,
                         "IC50_uM_VACV": [np.nan]*6})


def test_regressor_uses_zero_target_and_full_tail():
    data = frame(); y = np.array([0., 1., 2., 3., 4., 36066.667])
    model = RegressionV2("constant").fit(data, y)
    np.testing.assert_allclose(model.predict(data), y.mean())
    assert not {"SI VACV", "CC50_uM", "IC50_uM_VACV"} & set(allowed(data))


@pytest.mark.parametrize("target", [[-1.]*6, [np.nan]*6, [1.]])
def test_regressor_rejects_old_or_invalid_targets(target):
    with pytest.raises(ValueError):
        RegressionV2("constant").fit(frame(), target)


def test_measured_10000_is_valid_regression_target():
    data = frame()
    model = RegressionV2("constant").fit(data, np.full(len(data), 10000.))
    np.testing.assert_array_equal(model.predict(data), np.full(len(data), 10000.))


def test_kernel_predictions_ignore_forbidden_fields_and_query_target():
    data = frame(); y = np.arange(6.)
    model = RegressionV2("kernel-raw").fit(data, y)
    first = model.predict(data)
    data["SI VACV"] = 1e10
    data["CC50_uM"] = 0.
    data["IC50_uM_VACV"] = 0.
    np.testing.assert_allclose(first, model.predict(data), rtol=0, atol=0)
    assert np.isfinite(first).all() and (first >= 0).all()
