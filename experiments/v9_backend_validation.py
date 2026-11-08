"""Predeclared numeric model-family comparison and provisional data ablation."""
import argparse
from collections.abc import Mapping
import importlib.metadata
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.nested_selection import evaluate_outer_fold, select_weights
from experiments.v2_analog import diagnostics, fingerprint, tanimoto
from experiments.v2_context_analog import contexts, same_context
from experiments.v4_pair_delta import fit_predict
from experiments.v8_nested_validation import sha, grouping, make_folds, versions
from experiments.v9_model_backends import predict_backends, BACKEND_CONFIG
from vacv.dataset import load
from vacv.curate import standardize
from vacv.pipeline import structure_matrix, scaffolds_of

MODELS = ("v4", "et-raw", "et-log", "cat-raw", "cat-log", "kernel-raw", "kernel-log", "et-log-external")
PROTOCOL = Path("docs/v9-backend-protocol.json")
OUTPUT = Path("experiments/results/v9-backend-validation")
SNAPSHOT = Path("experiments/results/v9-similar-snapshot/provisional-exact.csv")
V8 = Path("docs/v8-nested-protocol.json")
SOURCES = tuple(map(Path, ("experiments/v9_backend_validation.py", "experiments/v9_model_backends.py",
    "experiments/v9_similar_snapshot.py", "experiments/nested_selection.py",
    "experiments/v8_nested_validation.py", "experiments/v4_pair_delta.py",
    "experiments/v2_analog.py", "experiments/v2_context_analog.py",
    "vacv/dataset.py", "vacv/features.py", "vacv/pipeline.py", "vacv/curate.py")))
INPUTS = (V8, SNAPSHOT, SNAPSHOT.with_name("manifest.json"),
          Path("data/raw_v2/train_v2.csv"), Path("data/raw_v2/test.csv"))


def plain(value):
    return {k: plain(v) for k, v in value.items()} if isinstance(value, Mapping) else value


def config():
    return {"models": list(MODELS), "backend_config": plain(BACKEND_CONFIG),
            "primary_endpoint": "core-blend vs v4 on identity pooled OOF raw MSE",
            "selection": "inner GroupKFold3 only; individual or convex half-grid",
            "core_candidates": list(MODELS[:-1]), "exploratory_candidates": list(MODELS),
            "grid_units": 2, "external_total_weight_per_identity": .25,
            "external_status": "unverified conditions; provisional exploratory ablation only",
            "external_scaffold_policy": "exclude external matching any outer OR current inner validation nonempty scaffold",
            "confirmation": "core-blend improves primary and both identity repeats, scaffold no worse",
            "all_targets_retained": True, "features": "existing structure_matrix; kernels squared binary similarity",
            "test_inference": False, "upload": False,
            "limitation": "same historically reused dataset and pre-existing outer folds; not a fresh holdout"}


def freeze():
    data = load(Path("data/raw_v2"), require_v2=True)
    identities, groups = grouping(data.train)
    previous = json.loads(V8.read_text())
    assert previous["folds"] == make_folds(identities, groups)
    protocol = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config(),
                "versions": {**versions(), "catboost": importlib.metadata.version("catboost")},
                "sha256": {str(p): sha(p) for p in SOURCES + INPUTS},
                "folds": previous["folds"], "identities": identities.tolist(), "scaffold_groups": groups.tolist()}
    with PROTOCOL.open("x") as f:
        f.write(json.dumps(protocol, indent=2) + "\n")
    print("Frozen protocol", sha(PROTOCOL), flush=True)


def verify_groups(protocol, identities, groups):
    if (identities.tolist() != protocol["identities"]
            or groups.tolist() != protocol["scaffold_groups"]
            or make_folds(identities, groups) != protocol["folds"]):
        raise ValueError("Identity groups, scaffold groups or folds differ from frozen protocol")


def run(expected):
    if sha(PROTOCOL) != expected:
        raise ValueError("Protocol hash mismatch")
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["config"] == config()
    assert protocol["versions"] == {**versions(), "catboost": importlib.metadata.version("catboost")}
    for p, digest in protocol["sha256"].items():
        if sha(p) != digest:
            raise ValueError(f"Frozen source/input changed: {p}")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = load(Path("data/raw_v2"), require_v2=True)
    external = pd.read_csv(SNAPSHOT)
    info = json.loads(SNAPSHOT.with_name("manifest.json").read_text())
    assert sha(SNAPSHOT) == info["snapshot_sha256"]
    ids, groups = grouping(data.train)
    # Test rows are not split: the curation helper rejects one out-of-range row.
    # Block all available standardized keys AND all exact competition strings.
    test_keys = [standardize(s).inchikey for s in data.test.SMILES]
    test_ids = {k[:14] for k in test_keys if k}
    ext_ids, _ = grouping(external)
    assert not set(ext_ids) & (set(ids) | set(test_ids))
    assert not set(external.SMILES) & set(pd.concat([data.train.SMILES, data.test.SMILES]))
    assert len(external) == info["provisional_rows"]
    np.testing.assert_array_equal(ext_ids, external.key)
    verify_groups(protocol, ids, groups)
    x, xe = structure_matrix(data.train), structure_matrix(external)
    bits, y = fingerprint(data.train), data.y
    sim = tanimoto(bits, bits)
    kernel = sim ** 2
    ctx = contexts(data.train)
    match = same_context(ctx, ctx)
    scaffold, ext_scaffold = np.asarray(scaffolds_of(data.train)), np.asarray(scaffolds_of(external))
    names = (*MODELS, "core-single", "core-blend", "exploratory-single", "exploratory-blend")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    selections, metrics, external_audit = [], [], []
    for scheme, folds in protocol["folds"].items():
        oof = np.full((len(y), len(names)), np.nan)
        assignment = np.full(len(y), -1)
        for fold, indices in enumerate(folds):
            tr, va = np.asarray(indices["train"]), np.asarray(indices["valid"])
            inner = [(np.asarray(f["train"]), np.asarray(f["valid"])) for f in indices["inner"]]
            blocked_scaffolds = set(scaffold[va]) - {""} if scheme == "scaffold" else set()

            def predict(fit, query, labels):
                blocked = blocked_scaffolds | (set(scaffold[query]) - {""} if scheme == "scaffold" else set())
                keep = ~np.isin(ext_scaffold, list(blocked))
                external_audit.append({"scheme": scheme, "fold": fold, "fit_rows": fit.tolist(),
                    "query_rows": query.tolist(), "external_rows": np.flatnonzero(keep).tolist()})
                predictions = predict_backends(x[fit], labels, x[query], kernel[np.ix_(fit, fit)],
                    kernel[np.ix_(query, fit)], external_x=xe[keep], external_y=external.target.to_numpy()[keep],
                    external_weight=external.weight.to_numpy()[keep])
                masked = np.full(len(y), np.nan)
                masked[fit] = labels
                baseline = fit_predict("ridge-log", bits, masked, sim, match, fit, query)[0]
                return np.column_stack([baseline, *[predictions[n] for n in MODELS[1:]]])

            result = evaluate_outer_fold(y, tr, va, inner, predict, len(MODELS), units=2)
            core = select_weights(result["inner_oof"][:, :-1], y[tr], units=2)
            base = result["base"]
            oof[va] = np.column_stack([base, base[:, core["single_index"]],
                base[:, :-1] @ np.asarray(core["weights"]), result["single"], result["blend"]])
            assignment[va] = fold
            selections.append({"scheme": scheme, "fold": fold, "core": core, "exploratory": result["selection"]})
            pd.DataFrame({"row": tr, "target": y[tr], **dict(zip(MODELS, result["inner_oof"].T))}).to_csv(
                OUTPUT / f"inner-{scheme}-{fold}.csv", index=False)
            (OUTPUT / "selection-audit.json").write_text(json.dumps(selections, indent=2) + "\n")
            (OUTPUT / "external-fit-audit.json").write_text(json.dumps(external_audit, indent=2) + "\n")
            print(scheme, fold, "core", MODELS[core["single_index"]], "blend", core["weights"], flush=True)
        assert np.isfinite(oof).all() and (assignment >= 0).all()
        pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": assignment,
                      **dict(zip(names, oof.T))}).to_csv(OUTPUT / f"oof-{scheme}.csv", index=False)
        metrics += [{"split": scheme, "model": n, **diagnostics(y, p)} for n, p in zip(names, oof.T)]
        (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(scheme, [(m["model"], round(m["mse"])) for m in metrics[-len(names):]], flush=True)
    score = {(m["split"], m["model"]): m["mse"] for m in metrics}
    manifest = {"protocol_sha256": expected, "finished_at": datetime.now(timezone.utc).isoformat(),
                "confirmation_passed": all(score[s, "core-blend"] < score[s, "v4"] for s in
                    ("identity", "identity71", "identity72")) and score["scaffold", "core-blend"] <= score["scaffold", "v4"],
                "external_candidate_promotable": False, "test_predictions": False, "uploaded": False,
                "test_identity_standardization_rejections": sum(k is None for k in test_keys),
                "sha256": {p.name: sha(p) for p in sorted(OUTPUT.iterdir())}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--protocol-sha256")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        freeze() if args.freeze else run(args.protocol_sha256)
