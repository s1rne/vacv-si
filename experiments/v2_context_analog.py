"""Predeclared context-gated analog experts, with structural log-KNN fallback."""
import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from vacv.dataset import load, submission
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = {"context-raw70": (.70, False), "context-raw85": (.85, False),
            "context-log70": (.70, True)}


def contexts(frame):
    return frame[["strain", "Culture"]].fillna("").astype(str).apply(lambda c: c.str.strip().str.casefold()).to_numpy()


def same_context(query, train):
    known_q = (query != "").all(axis=1)
    known_t = (train != "").all(axis=1)
    return (query[:, None, :] == train[None, :, :]).all(axis=2) & known_q[:, None] & known_t[None, :]


def feature_coverage(similarity, context_match):
    return {"rows": len(similarity), "any_known_matching_context": int(context_match.any(axis=1).sum()),
            "same_context_sim70": int((context_match & (similarity >= .70)).any(axis=1).sum()),
            "same_context_sim85": int((context_match & (similarity >= .85)).any(axis=1).sum()),
            "any_structure_sim70": int((similarity >= .70).any(axis=1).sum()),
            "any_structure_sim85": int((similarity >= .85).any(axis=1).sum())}


def predict(similarity, context_match, y, profile):
    threshold, log = PROFILES[profile]
    predictions, fired, counts = [], [], []
    for s, same in zip(similarity, context_match):
        candidates = np.flatnonzero(same & (s >= threshold))
        use_expert = len(candidates) > 0
        if use_expert:
            ix = candidates[np.argsort(-s[candidates], kind="stable")[:3]]
            target = np.log1p(y[ix]) if log else y[ix]
        else:
            candidates = np.flatnonzero(s >= 0)
            ix = candidates[np.argsort(-s[candidates], kind="stable")[:8]]
            target = np.log1p(y[ix])
        weights = s[ix] ** 8
        value = float(weights @ target / weights.sum()) if weights.sum() > 0 else float(np.median(target))
        predictions.append(np.expm1(value) if log or not use_expert else value)
        fired.append(use_expert)
        counts.append(len(ix) if use_expert else 0)
    return np.maximum(0, predictions), np.asarray(fired), np.asarray(counts)


def coverage(y, fired, counts):
    tail = y >= 1000
    return {"fired": int(fired.sum()), "total": len(y), "fraction": float(fired.mean()),
            "tail_fired": int(fired[tail].sum()), "tail_total": int(tail.sum()),
            "max_label_covered": bool(fired[np.argmax(y)]),
            "expert_neighbor_mean": float(counts[fired].mean()) if fired.any() else 0.}


def production_baseline():
    """Exact current production ET on paired folds; per-fold predictions cached."""
    from vacv.regression_v2 import RegressionV2
    data = load(Path("data/raw_v2"), require_v2=True)
    out = Path("experiments/results/v2-context-analog/production-baseline")
    out.mkdir(parents=True, exist_ok=True)
    metrics = []
    for split in (12, 13, 14, "scaffold"):
        if split == "scaffold":
            scaffold = scaffold_folds(scaffolds_of(data.train), 5)
            iterator = (fold_indices(scaffold, f) for f in range(5))
        else:
            iterator = KFold(5, shuffle=True, random_state=split).split(data.y)
        pieces = []
        for fold, (tr, va) in enumerate(iterator):
            path = out/f"split-{split}-fold-{fold}.csv"
            if path.exists():
                frame = pd.read_csv(path)
                if not np.array_equal(frame.row.to_numpy(), va) or not np.allclose(frame.target, data.y[va]):
                    raise ValueError("Baseline cache row/target mismatch")
            else:
                model = RegressionV2("et-blend-log").fit(data.train.iloc[tr], data.y[tr])
                pred = model.predict(data.train.iloc[va])
                frame = pd.DataFrame({"row": va, "target": data.y[va], "fold": fold, "et-blend-log": pred})
                frame.to_csv(path, index=False)
            pieces.append(frame)
            print(f"production baseline split={split} fold={fold}", flush=True)
        frame = pd.concat(pieces).sort_values("row")
        frame.to_csv(out/f"oof-{split}.csv", index=False)
        metrics.append({"split": split, "model": "et-blend-log", **diagnostics(data.y, frame["et-blend-log"].to_numpy())})
        (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
        print(metrics[-1], flush=True)
    manifest = {"train_sha256": hashlib.sha256(Path("data/raw_v2/train_v2.csv").read_bytes()).hexdigest(),
                "production_source_sha256": hashlib.sha256(Path("vacv/regression_v2.py").read_bytes()).hexdigest(),
                "profile": "et-blend-log", "splits": [12, 13, 14, "scaffold"]}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def main():
    out = Path("experiments/results/v2-context-analog")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    y = data.y
    x, xt = fingerprint(data.train), fingerprint(data.test)
    sim, test_sim = tanimoto(x, x), tanimoto(xt, x)
    ctx, test_ctx = contexts(data.train), contexts(data.test)
    match, test_match = same_context(ctx, ctx), same_context(test_ctx, ctx)
    loo = sim.copy()
    np.fill_diagonal(loo, -1)
    coverages = {}
    for name in PROFILES:
        _, fired, counts = predict(loo, match, y, name)
        coverages[name] = coverage(y, fired, counts)
    (out/"coverage-loo.json").write_text(json.dumps(coverages, indent=2)+"\n")
    print("LOO coverage:", coverages, flush=True)
    metrics, winner = [], None
    with threadpool_limits(limits=2):
        for split in (12, 13, 14, "scaffold"):
            names = tuple(PROFILES) if split == 12 else (winner,)
            if split == "scaffold":
                scaffold = scaffold_folds(scaffolds_of(data.train), 5)
                iterator = (fold_indices(scaffold, f) for f in range(5))
            else:
                iterator = KFold(5, shuffle=True, random_state=split).split(y)
            pred = {n: np.full(len(y), np.nan) for n in names}
            fallback = np.full(len(y), np.nan)
            gates = {n: np.zeros(len(y), dtype=bool) for n in names}
            counts = {n: np.zeros(len(y), dtype=int) for n in names}
            folds = np.full(len(y), -1)
            for fold, (tr, va) in enumerate(iterator):
                fallback[va] = predict(sim[np.ix_(va, tr)], np.zeros((len(va), len(tr)), dtype=bool), y[tr], "context-log70")[0]
                for name in names:
                    p, fired, number = predict(sim[np.ix_(va, tr)], match[np.ix_(va, tr)], y[tr], name)
                    pred[name][va], gates[name][va], counts[name][va] = p, fired, number
                folds[va] = fold
            for name, p in pred.items():
                assert np.isfinite(p).all()
                metrics.append({"split": split, "model": name, **diagnostics(y, p), "coverage": coverage(y, gates[name], counts[name]),
                                "fallback_control_mse": float(np.mean((fallback-y)**2))})
            extra = {f"{n}-gate": g for n, g in gates.items()}
            pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds, "structure-log-control": fallback, **pred, **extra}).to_csv(out/f"oof-{split}.csv", index=False)
            (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            print(metrics[-len(names):], flush=True)
            if split == 12:
                winner = min(metrics, key=lambda item: item["mse"])["model"]
        p, fired, counts = predict(test_sim, test_match, y, winner)
        template = pd.read_csv("data/raw_v2/sample_submit.csv")
        result = submission(data.test["index"], p).set_index("index").loc[template["index"]].reset_index()
        result.to_csv(out/f"submission-{winner}.csv", index=False)
    manifest = {"profiles": PROFILES, "discovery": 12, "confirmation": [13, 14], "winner": winner,
                "external_data": False, "target_version": "train_v2", "test_gate_fired": int(fired.sum()),
                "feature_coverage_train_loo": feature_coverage(loo, match & (loo >= 0)),
                "feature_coverage_test": feature_coverage(test_sim, test_match),
                "train_sha256": hashlib.sha256(Path("data/raw_v2/train_v2.csv").read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "helper_script_sha256": hashlib.sha256(Path("experiments/v2_analog.py").read_bytes()).hexdigest()}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-baseline", action="store_true")
    args = parser.parse_args()
    production_baseline() if args.production_baseline else main()
