import numpy as np
import pandas as pd
import pytest

from vacv.foundation import compact_features
from vacv.postprocess import conditional_lookup
from vacv.source_kernel import SourceKernelClassifier
from vacv.transductive import HarmonicSelectivityModel, graph_weights


def frame():
    return pd.DataFrame({"SMILES": ["CCO", "CCN", "CCC", "CCCO", "CCCN", "CCCC"],
                         "strain": ["a", "a", "a", "b", "b", "b"],
                         "Culture": ["cell"] * 6, "Title": list("abcdef"),
                         "matched": [0., 1., 0., 1., 0., 0.]})


@pytest.mark.parametrize("model", [HarmonicSelectivityModel(),
                                  HarmonicSelectivityModel(evidence_strength=0.2),
                                  SourceKernelClassifier()])
def test_models_never_read_validation_or_unlabeled_targets(model):
    x = frame()
    train, valid, test = x.iloc[:4], x.iloc[4:], x.iloc[4:].copy()
    y = np.array([1., 10000., 2., 10000.])
    p, prediction = model.fit_predict(train, y, valid, test)
    poisoned = valid.assign(**{"SI VACV": [10000., 10000.], "CC50_uM": [99., 0.],
                              "IC50_uM_VACV": [np.nan, np.nan]})
    poisoned_test = test.assign(**{"SI VACV": [999999., -20.], "CC50_uM": [1., 0.]})
    p2, prediction2 = model.fit_predict(train, y, poisoned, poisoned_test)
    assert np.allclose(p, p2)
    assert np.allclose(prediction, prediction2)
    assert np.isfinite(prediction).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_graph_is_symmetric_and_has_no_self_edges():
    weights = graph_weights(frame())
    assert np.allclose(weights, weights.T)
    assert np.all(np.diag(weights) == 0)
    assert (weights >= 0).all()


def test_foundation_features_whitelist_columns_and_preserve_invalid_rows():
    x = frame()
    x.loc[0, "SMILES"] = "invalid(smiles"
    features = compact_features(x)
    changed = compact_features(x.assign(**{"SI VACV": [10000.] * 6, "CC50_uM": [999.] * 6}))
    pd.testing.assert_frame_equal(features, changed)
    assert len(features) == len(x)
    assert "SI VACV" not in features


def test_conditional_lookup_respects_sentinel_uncertainty_and_absence():
    x = pd.DataFrame({"literature_si": [10., 10., 10., np.nan]})
    prediction = np.array([30., 40., 50., 60.])
    result = conditional_lookup(prediction, np.array([0., 0.5, 1., 0.9]), x)
    assert np.allclose(result, [10., 5005., 10000., 60.])
    assert np.array_equal(prediction, [30., 40., 50., 60.])


@pytest.mark.parametrize("p", [[-0.1], [1.1], [np.nan], [0.2, 0.3]])
def test_conditional_lookup_rejects_invalid_probabilities(p):
    with pytest.raises(ValueError):
        conditional_lookup([1.], p, pd.DataFrame({"literature_si": [10.]}))
