import numpy as np
import pandas as pd

from experiments.luna_v2_stable_extrapolation import features, stable_descriptor_features


def test_signed_log_and_quantile_clipping_bound_extreme_descriptor():
    train = np.array([[-2.0], [0.0], [2.0], [4.0], [6.0]])
    query = np.array([[1e16]])

    transformed, bounded = stable_descriptor_features(train, query)

    assert np.isfinite(transformed).all()
    assert np.isfinite(bounded).all()
    assert bounded[0, 0] == np.quantile(np.sign(train[:, 0]) * np.log1p(np.abs(train[:, 0])), 0.99)


def test_forbidden_competition_columns_do_not_change_features():
    train = pd.DataFrame({"SMILES": ["C", "CC", "CCC"], "Title": ["a", "b", "c"],
                          "strain": ["s1", "s1", "s2"], "Culture": ["x", "x", "y"]})
    query = pd.DataFrame({"SMILES": ["CCCC"], "Title": ["d"], "strain": ["s2"], "Culture": ["y"]})
    enriched_train = train.assign(CC50_uM=[1e9, 2e9, 3e9], IC50_uM_VACV=[1, 2, 3])
    enriched_query = query.assign(CC50_uM=[9e9], IC50_uM_VACV=[9])
    dx = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    dq = np.array([[1e16, -1e16]])
    embeddings = {smiles: np.array([i, i * 2.0, i % 2, 1.0])
                  for i, smiles in enumerate(["C", "CC", "CCC", "CCCC"], start=1)}

    baseline = features(train, query, dx, dq, embeddings)
    with_forbidden = features(enriched_train, enriched_query, dx, dq, embeddings)

    np.testing.assert_allclose(baseline[0], with_forbidden[0])
    np.testing.assert_allclose(baseline[1], with_forbidden[1])
