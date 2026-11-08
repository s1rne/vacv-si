"""Frozen nested evaluation of existing predictors; never generates submissions."""
import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.nested_selection import evaluate_outer_fold, simplex_grid
from experiments.v2_analog import diagnostics, fingerprint, tanimoto
from experiments.v2_context_analog import contexts, same_context, predict as analog
from experiments.v4_pair_delta import fit_predict
from vacv.checkpoint import load_embeddings
from vacv.curate import standardize
from vacv.dataset import load
from vacv.pipeline import scaffolds_of

MODELS = ("v4", "context-raw70", "embedding-pair", "mixed-pair")
PROTOCOL = Path("docs/v8-nested-protocol.json")
OUTPUT = Path("experiments/results/v8-nested-validation")
EMBEDDING = Path("data/raw/models/chemeleon_features.npz")
INPUTS = (Path("data/raw_v2/train_v2.csv"), EMBEDDING, EMBEDDING.with_suffix(".json"))
SOURCES = tuple(map(Path, (
    "experiments/v8_nested_validation.py", "experiments/nested_selection.py",
    "experiments/v2_analog.py", "experiments/v2_context_analog.py",
    "experiments/v4_pair_delta.py", "vacv/dataset.py", "vacv/pipeline.py",
    "vacv/checkpoint.py", "vacv/curate.py", "vacv/features.py", "vacv/units.py")))
CONFIG = {"models": list(MODELS), "baseline": "v4", "primary": "identity",
          "outer_schemes": {"identity": None, "identity71": 71, "identity72": 72, "scaffold": None},
          "inner_folds": 3, "inner_seed": 1700, "outer_folds": 5,
          "selection": "minimum pooled inner OOF raw MSE, all train rows",
          "simplex_units": 4, "tie": "exact numerical tie: first, descending lexicographic weight order",
          "endpoint": "nested-blend vs v4 primary pooled outer OOF raw MSE",
          "diagnostics": "all base models and nested-single; no outer model/weight selection",
          "confirmation": "blend improves primary and both repeats; scaffold does not worsen",
          "scaffold_groups": "connected components of shared nonempty scaffold OR standardized key14",
          "features": "existing v4 bits; train-only PCA32 seed0/scaler, embedding scale8 and mixed scale4",
          "targets": "official v2, all rows, no clipping or removal",
          "external_labels": False, "test_inference": False, "uploads": False,
          "limitation": "historically reused dataset; repeats are not independent holdouts"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    return {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "rdkit", "scipy")}


def connected_groups(identities, scaffolds):
    """Union both relations, preventing identity overlap even across scaffolds."""
    parent = list(range(len(identities)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, (identity, scaffold) in enumerate(zip(identities, scaffolds, strict=True)):
        for key in [("identity", identity)] + ([("scaffold", scaffold)] if scaffold else []):
            if key in seen:
                a, b = find(i), find(seen[key])
                parent[max(a, b)] = min(a, b)
            seen[key] = i
    return np.asarray([find(i) for i in range(len(parent))])


def make_folds(identities, scaffold_groups):
    schemes = {}
    for name, seed in CONFIG["outer_schemes"].items():
        groups = scaffold_groups if name == "scaffold" else identities
        splitter = GroupKFold(5, shuffle=seed is not None, random_state=seed)
        outer = []
        for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(groups)), groups=groups)):
            inside = GroupKFold(3, shuffle=True, random_state=1700 + fold)
            inner = [{"train": tr[a].tolist(), "valid": tr[b].tolist()} for a, b in
                     inside.split(np.zeros(len(tr)), groups=groups[tr])]
            outer.append({"train": tr.tolist(), "valid": va.tolist(), "inner": inner})
        schemes[name] = outer
    return schemes


def grouping(frame):
    identities = []
    for s in frame.SMILES:
        value = standardize(s)
        if not value.inchikey:
            raise ValueError("Identity standardization failed; cannot silently weaken grouping")
        identities.append(value.inchikey[:14])
    identities = np.asarray(identities)
    return identities, connected_groups(identities, scaffolds_of(frame))


def freeze():
    data = load(Path("data/raw_v2"), require_v2=True)
    identities, groups = grouping(data.train)
    protocol = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "config": CONFIG,
                "versions": versions(), "sha256": {str(p): sha(p) for p in INPUTS + SOURCES},
                "identity_groups": identities.tolist(), "scaffold_groups": groups.tolist(),
                "folds": make_folds(identities, groups),
                "weight_grid": simplex_grid(len(MODELS)).tolist()}
    with PROTOCOL.open("x") as f:
        f.write(json.dumps(protocol, indent=2) + "\n")
    print("Protocol frozen:", sha(PROTOCOL), flush=True)


def representation(bits, embedding, train):
    pca = PCA(n_components=min(32, len(train) - 1), svd_solver="randomized", random_state=0)
    ztrain = pca.fit_transform(embedding[train])
    scaler = StandardScaler().fit(ztrain)
    z = scaler.transform(pca.transform(embedding)) / np.sqrt(ztrain.shape[1])
    return bits, 8 * z, np.hstack([bits, 4 * z])


def run(expected_sha):
    if sha(PROTOCOL) != expected_sha:
        raise ValueError("Protocol hash mismatch")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["config"] != CONFIG or protocol["versions"] != versions():
        raise ValueError("Configuration or dependency version drift")
    for path, checksum in protocol["sha256"].items():
        if sha(path) != checksum:
            raise ValueError(f"Frozen input/source changed: {path}")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = load(Path("data/raw_v2"), require_v2=True)
    identities, groups = grouping(data.train)
    if (identities.tolist() != protocol["identity_groups"]
            or groups.tolist() != protocol["scaffold_groups"]
            or make_folds(identities, groups) != protocol["folds"]):
        raise ValueError("Frozen grouping or folds do not reproduce")
    mapping, _ = load_embeddings(EMBEDDING)
    embedding = np.stack([mapping[s] for s in data.train.SMILES])
    bits, y = fingerprint(data.train), data.y
    sim = tanimoto(bits, bits)
    ctx = contexts(data.train)
    match = same_context(ctx, ctx)

    def predict(tr, va, fit_targets):
        # Even accidental access outside the fit subset returns NaN, never a heldout label.
        labels = np.full(len(y), np.nan)
        labels[tr] = fit_targets
        reps = representation(bits, embedding, tr)
        base = fit_predict("ridge-log", reps[0], labels, sim, match, tr, va)[0]
        context = analog(sim[np.ix_(va, tr)], match[np.ix_(va, tr)], fit_targets, "context-raw70")[0]
        others = [fit_predict("ridge-log", x, labels, sim, match, tr, va)[0] for x in reps[1:]]
        return np.column_stack([base, context, *others])

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    metrics, selections = [], []
    names = (*MODELS, "nested-single", "nested-blend")
    for scheme, outer in protocol["folds"].items():
        oof = np.full((len(y), len(names)), np.nan)
        assignment = np.full(len(y), -1)
        scheme_groups = groups if scheme == "scaffold" else identities
        for fold, indices in enumerate(outer):
            tr, va = np.asarray(indices["train"]), np.asarray(indices["valid"])
            inner = [(np.asarray(f["train"]), np.asarray(f["valid"])) for f in indices["inner"]]
            for a, b in [(tr, va), *inner]:
                if set(identities[a]) & set(identities[b]) or set(scheme_groups[a]) & set(scheme_groups[b]):
                    raise ValueError("Group leakage in frozen split")
            result = evaluate_outer_fold(y, tr, va, inner, predict, len(MODELS))
            oof[va] = np.column_stack([result["base"], result["single"], result["blend"]])
            assignment[va] = fold
            selected = {"scheme": scheme, "fold": fold, **result["selection"]}
            selections.append(selected)
            pd.DataFrame({"row": tr, "target": y[tr], **dict(zip(MODELS, result["inner_oof"].T))}).to_csv(
                OUTPUT / f"inner-{scheme}-{fold}.csv", index=False)
            (OUTPUT / "selection-audit.json").write_text(json.dumps(selections, indent=2) + "\n")
            print(scheme, fold, "single", MODELS[selected["single_index"]], "weights", selected["weights"], flush=True)
        if not np.isfinite(oof).all() or (assignment < 0).any():
            raise ValueError("Incomplete outer OOF")
        pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": assignment,
                      **dict(zip(names, oof.T))}).to_csv(OUTPUT / f"oof-{scheme}.csv", index=False)
        metrics += [{"split": scheme, "model": name, **diagnostics(y, p)} for name, p in zip(names, oof.T)]
        (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(scheme, [(m["model"], round(m["mse"])) for m in metrics[-len(names):]], flush=True)
    score = {(m["split"], m["model"]): m["mse"] for m in metrics}
    improved = {s: score[s, "nested-blend"] < score[s, "v4"] for s in protocol["folds"]}
    manifest = {"protocol_sha256": expected_sha, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "primary_blend_improved": improved["identity"], "blend_improved": improved,
                "confirmation_passed": all(improved[s] for s in ("identity", "identity71", "identity72"))
                and score["scaffold", "nested-blend"] <= score["scaffold", "v4"],
                "submission_created": False, "uploaded": False,
                "artifacts_sha256": {p.name: sha(p) for p in sorted(OUTPUT.iterdir())}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--freeze", action="store_true")
    choice.add_argument("--protocol-sha256")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        freeze() if args.freeze else run(args.protocol_sha256)
