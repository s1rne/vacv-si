"""Fold isolation and endpoint semantics of the optional embedding experiment."""

import numpy as np
import pandas as pd
import pytest
import hashlib
import json
import sys
from types import SimpleNamespace

import experiments.pretrained_embeddings as experiment
from experiments.pretrained_embeddings import FoldEmbedding, categorical_kernel, predict_head, tabular
from vacv.literature import PRESENCE_FEATURES
from vacv.postprocess import conditional_lookup


def frame(n=5):
    result = pd.DataFrame({"strain": ["s"] * n, "Culture": ["c"] * n,
                           "Title": ["1a"] * n, "matched": np.zeros(n),
                           "SMILES": ["CC"] * n})
    for name in PRESENCE_FEATURES:
        result[name] = np.zeros(n)
    return result


def test_pca_and_bandwidth_do_not_fit_validation():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(12, 8))
    reducer = FoldEmbedding(4).fit(train)
    mean = reducer.pca_.mean_.copy()
    gamma = reducer.gamma_
    reducer.transform(np.full((3, 8), 1e6))
    np.testing.assert_allclose(mean, train.mean(axis=0))
    np.testing.assert_array_equal(reducer.pca_.mean_, mean)
    assert reducer.gamma_ == gamma


def test_kernel_uses_group_and_evidence_without_targets():
    tr = frame(2)
    va = frame(2)
    va.loc[0, "matched"] = 1
    va.loc[1, "strain"] = "new"
    np.testing.assert_array_equal(categorical_kernel(va, tr), [[2, 2], [0, 0]])


@pytest.mark.parametrize("profile", ["che-ridge", "che-additive", "che-svr", "che-tabicl"])
def test_single_active_class_includes_real_high_si(profile):
    tr = frame(3)
    p = predict_head(tr, np.array([36066.667, 1, 2]), frame(2), np.zeros((3, 4)),
                     np.zeros((2, 4)), profile, None)
    np.testing.assert_array_equal(p, [0, 0])


@pytest.mark.parametrize("profile", ["che-ridge", "che-additive", "che-svr"])
def test_heads_return_finite_probabilities(profile):
    rng = np.random.default_rng(42)
    p = predict_head(frame(5), np.array([10000, 0, 1, 10000, 36066.667]), frame(2),
                     rng.normal(size=(5, 8)), rng.normal(size=(2, 8)), profile, None)
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_tabular_features_have_explicit_allowlist():
    x = frame(2)
    x["SI VACV"] = [10000, 1]
    x["CC50_uM"] = [12, 13]
    x["IC50_uM_VACV"] = [10, 11]
    transformed = tabular(x, np.ones((2, 3)))
    assert not {"SI VACV", "CC50_uM", "IC50_uM_VACV"} & set(transformed.columns)


def test_soft_lookup_uses_final_blended_probability():
    enriched = pd.DataFrame({"literature_si": [100., np.nan]})
    p = 0.75 * np.array([.4, .8]) + 0.25 * np.array([.2, .6])
    raw = p * 10000 + (1-p) * 42
    result = conditional_lookup(raw, p, enriched)
    np.testing.assert_allclose(result, [p[0]*10000+(1-p[0])*100, raw[1]])


def test_cv_export_without_train_index_and_soft_ensemble(tmp_path, monkeypatch):
    enriched = frame(10)
    enriched["literature_si"] = 100.
    y = np.tile([10000., 42.], 5)
    features = tmp_path / "features.npz"
    np.savez(features, smiles=np.array(["CC"]), embeddings=np.ones((1, 3)))
    features.with_suffix(".json").write_text(json.dumps({
        "embedding_sha256": hashlib.sha256(features.read_bytes()).hexdigest()}))
    monkeypatch.setattr(experiment, "load", lambda: SimpleNamespace(train=enriched, y=y))
    monkeypatch.setattr(experiment, "attach", lambda x: x)
    monkeypatch.setattr(experiment, "predict_head", lambda *a: np.full(len(a[2]), .2))
    monkeypatch.setattr(experiment, "predict_ensemble", lambda tr, y, va, *a: (
        np.full(len(va), .4), np.full(len(va), .4*10000+.6*42), np.full(len(va), 100.)))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["experiment", "--features", str(features),
                                      "--profiles", "che-ridge", "--splits", "random-0",
                                      "--soft", "--out", str(out)])
    experiment.main()
    oof = pd.read_csv(out / "random-0-hybrid+che-ridge-soft.csv")
    np.testing.assert_array_equal(oof["index"], np.arange(10))
    np.testing.assert_allclose(oof.probability, .35)
    np.testing.assert_allclose(oof.prediction, .35*10000+.65*100)
    assert len(json.loads((out/"metrics.json").read_text())) == 6
