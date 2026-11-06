import numpy as np
import pandas as pd
import pytest

from vacv.graph_correction import (
    GraphResidualCorrector, correct_probability, crossfit_probabilities, propagate_residuals,
)
from vacv.transductive import graph_weights
from vacv.postprocess import conditional_lookup


def frame(n=12):
    return pd.DataFrame({"SMILES": ["CCO", "CCN", "CCC"] * (n // 3),
                         "strain": ["a"] * n, "Culture": ["cells"] * n,
                         "Title": [str(i) for i in range(n)], "matched": [0.] * n,
                         "literature_si": [np.nan] * n})


def test_diffusion_reaches_two_hop_node_and_onehop_does_not():
    graph = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    assert np.allclose(propagate_residuals(graph, np.array([1.]), "onehop", 1.), [.5, 0.])
    assert np.allclose(propagate_residuals(graph, np.array([1.]), "diffusion", 1.), [.4, .2])


def test_zero_errors_and_disconnected_nodes_have_zero_correction():
    graph = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
    for mode in ["onehop", "diffusion"]:
        assert np.array_equal(propagate_residuals(graph, np.zeros(1), mode), np.zeros(2))
        assert propagate_residuals(graph, np.array([-.5]), mode)[1] == 0.


def test_crossfit_own_target_and_outer_validation_targets_are_unavailable(monkeypatch):
    import vacv.graph_correction as correction
    full = frame(15)
    outer_train = full.iloc[:12].assign(**{"SI VACV": [-999.] * 12, "CC50_uM": [1.] * 12})
    outer_valid = full.iloc[12:]
    y = np.array([1., 10000.] * 6)
    calls = []

    def predictor(train, target, query, seeds, profile):
        assert "SI VACV" not in train and "SI VACV" not in query
        assert "CC50_uM" not in train and "CC50_uM" not in query
        assert set(train.index).isdisjoint(outer_valid.index)
        assert set(train.index).isdisjoint(query.index)
        assert np.array_equal(target, y[train.index])
        calls.append((train.index.tolist(), query.index.tolist()))
        p = np.full(len(query), np.mean(target == 10000))
        return p, p * 10000, p * 10000

    monkeypatch.setattr(correction, "predict_ensemble", predictor)
    original = crossfit_probabilities(outer_train, y)
    y[0] = 10000.
    changed = crossfit_probabilities(outer_train, y)
    assert original[0] == changed[0]
    assert len(calls) == 8


def test_corrector_whitelists_features_and_preserves_literature_override(monkeypatch):
    import vacv.graph_correction as correction
    seen_columns = []

    def predictor(train, target, query, seeds, profile):
        seen_columns.extend([list(train.columns), list(query.columns)])
        p = np.full(len(query), .25)
        return p, p * 10000, p * 10000

    monkeypatch.setattr(correction, "predict_ensemble", predictor)
    train = frame()
    y = np.array([36066., 10000.] * 6)
    estimator = GraphResidualCorrector().fit(train, y)
    assert estimator.active_mean_ == 36066.
    assert estimator.residuals_[0] == -.25
    query = frame(3).assign(**{"SI VACV": [999.] * 3, "CC50_uM": [np.nan] * 3,
                              "IC50_uM_VACV": [0.] * 3})
    query.loc[0, "literature_si"] = 42.
    components = estimator.components(query)
    probability, raw, final = estimator.predictions(query, components, .25, "diffusion")
    assert final[0] == 42.
    assert np.isfinite(raw).all() and ((probability >= 0) & (probability <= 1)).all()
    assert all(not {"SI VACV", "CC50_uM", "IC50_uM_VACV"} & set(cols) for cols in seen_columns)


def test_graph_geometry_ignores_targets_and_invalid_molecule_is_finite():
    query = frame(3)
    query.loc[0, "SMILES"] = "bad(smiles"
    weights = graph_weights(query)
    altered = query.assign(**{"SI VACV": [1., 2., 3.], "IC50_uM_VACV": [np.nan, 0., 1.]})
    assert np.allclose(weights, graph_weights(altered))
    assert np.isfinite(weights).all() and np.allclose(weights, weights.T)
    assert np.all(np.diag(weights) == 0)


@pytest.mark.parametrize("penalty", [0., -1., np.nan])
def test_residual_solver_rejects_invalid_regularization(penalty):
    with pytest.raises(ValueError):
        propagate_residuals(np.array([[0., 1.], [1., 0.]]), np.array([.2]), regularization=penalty)


def test_output_guard_preserves_previous_run(tmp_path, monkeypatch):
    from experiments.graph_correction import main
    existing = tmp_path / "metrics.json"
    existing.write_text("preserve")
    monkeypatch.setattr("sys.argv", ["graph_correction", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and existing.read_text() == "preserve"


def test_confidence_gate_is_zero_at_certainty_and_one_at_maximum_uncertainty():
    p = np.array([0., .5, 1.])
    delta = np.array([.3, .3, -.3])
    assert np.allclose(correct_probability(p, delta, .5, True), [0., .65, 1.])
    assert np.allclose(correct_probability(p, delta, 0., True), p)
    assert np.allclose(correct_probability(p, delta, .5, False), [.15, .65, .85])


def test_soft_lookup_combines_activity_uncertainty_and_preserves_unmatched_raw():
    query = pd.DataFrame({"literature_si": [42., 42., np.nan, 20000.]})
    result = conditional_lookup(np.array([50., 5000., 123., 9000.]),
                                np.array([0., .5, .7, .2]), query)
    assert np.allclose(result, [42., 5021., 123., 10000.])


def test_soft_lookup_rejects_invalid_probability():
    with pytest.raises(ValueError):
        conditional_lookup([20.], [1.1], pd.DataFrame({"literature_si": [42.]}))
