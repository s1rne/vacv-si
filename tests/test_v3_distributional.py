import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments import v3_distributional as module


def frame():
    return pd.DataFrame({"SMILES": ["C", "CC", "CCC", "N", "CN", "CO"],
        "Title": list("abcdef"), "strain": ["VACV"]*6, "Culture": ["Vero"]*6})


def test_inner_residuals_use_only_other_training_rows(monkeypatch):
    y = np.array([0., 1., 2., 3., 4., 20.])
    monkeypatch.setattr(module, "fit_mean", lambda f, z, trees: float(z.mean()))
    monkeypatch.setattr(module, "mean_predict", lambda fitted, f: np.full(len(f), fitted))
    observed = module.crossfit_residuals(frame(), y, trees=2)
    expected = np.empty(len(y))
    for train, valid in KFold(3, shuffle=True, random_state=42).split(y):
        expected[valid] = (y[valid]-y[train].mean())**2
    np.testing.assert_array_equal(observed, expected)


def test_query_answers_and_forbidden_fields_cannot_affect_prediction():
    f = frame()
    model = module.DistributionalRegression(trees=5).fit(f, np.array([0, 1, 2, 3, 4, 36066.667]))
    before = model.predict_all(f)
    f["SI VACV"] = 1e20
    f["CC50_uM"] = 1e20
    f["IC50_uM_VACV"] = -99
    after = model.predict_all(f)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
        assert np.isfinite(after[name]).all()
    assert (after["variance25"] >= after["log-baseline"]).all()
