"""Paired interval-aware external SI experiment, requiring an audited snapshot.

Snapshot schema: SMILES, lower_si, upper_si, kind, optional source_id.
kind is exact/lower_bound/upper_bound. A bound is never an exact target.
The caller must audit endpoint provenance before supplying this snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from vacv.curate import standardize
from vacv.dataset import load, submission
from vacv.interval_regression import IntervalKernelRegressor
from vacv.pipeline import structure_matrix, scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = ("competition-only", "external-exact", "external-intervals")
FIELDS = {"SMILES", "lower_si", "upper_si", "kind", "source_id"}


def key(smiles):
    value = standardize(str(smiles)).inchikey
    return value[:14] if value else None


def read_snapshot(path, train, test):
    frame = pd.read_csv(path, usecols=lambda c: c in FIELDS)
    required = FIELDS-{"source_id"}
    if not required <= set(frame):
        raise ValueError(f"Audited snapshot must contain {sorted(required)}")
    lower, upper = frame.lower_si.to_numpy(float), frame.upper_si.to_numpy(float)
    if (np.isnan(lower).any() or np.isnan(upper).any() or (lower < 0).any()
            or (upper < lower).any() or np.isposinf(lower).any()):
        raise ValueError("Invalid nonnegative SI intervals")
    exact = frame.kind.eq("exact") & np.isfinite(upper) & (lower == upper)
    lower_bound = frame.kind.eq("lower_bound") & np.isfinite(lower) & np.isposinf(upper)
    upper_bound = frame.kind.eq("upper_bound") & (lower == 0) & np.isfinite(upper)
    if not (exact | lower_bound | upper_bound).all():
        raise ValueError("Interval kind must agree with endpoints")
    competition = pd.concat([train.SMILES, test.SMILES]).astype(str)
    blocked_smiles = set(competition)
    blocked_keys = {key(s) for s in competition}
    frame["key14"] = [key(s) for s in frame.SMILES]
    bad = frame.key14.isna() | frame.key14.isin(blocked_keys) | frame.SMILES.isin(blocked_smiles)
    excluded = frame.loc[bad].copy()
    frame = frame.loc[~bad].copy()
    # Avoid silently averaging inconsistent assays/intervals for one molecule.
    duplicates = frame.key14.duplicated(keep=False)
    excluded = pd.concat([excluded, frame.loc[duplicates]])
    frame = frame.loc[~duplicates].reset_index(drop=True)
    if not len(frame):
        raise ValueError("No disjoint unique audited external molecules")
    return frame, excluded


def fingerprint(frame):
    return (structure_matrix(frame)[:, :2048] > 0).astype(float)


def tanimoto(a, b):
    dot = a@b.T
    denominator = a.sum(1)[:, None]+b.sum(1)[None, :]-dot
    return np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-external", type=Path, required=True,
                        help="Explicitly audited SI interval snapshot, never a raw database dump")
    parser.add_argument("--out", type=Path, default=Path("experiments/results/v3-intervals"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    data = load(Path("data/raw_v2"), require_v2=True)
    external, excluded = read_snapshot(args.approved_external, data.train, data.test)
    args.out.mkdir(parents=True)
    external.to_csv(args.out/"external-training.csv", index=False)
    excluded.to_csv(args.out/"external-excluded.csv", index=False)
    y = data.y
    features = np.vstack([fingerprint(data.train), fingerprint(external)])
    kernel = tanimoto(features, features)
    external_indices = np.arange(len(y), len(features))
    exact_indices = external_indices[external.kind.eq("exact").to_numpy()]
    lower = np.r_[np.log1p(y), np.log1p(external.lower_si.to_numpy(float))]
    upper = np.r_[np.log1p(y), np.log1p(external.upper_si.to_numpy(float))]
    weights = np.r_[np.ones(len(y)), np.full(len(external), .25)]
    manifest = {"profiles": PROFILES, "alpha": 1., "external_weight": .25,
                "discovery": 21, "confirmation": [22, 23], "target_version": "train_v2",
                "train_rows": len(y), "external_rows": len(external),
                "external_exact": int(external.kind.eq("exact").sum()),
                "external_bounds": int(external.kind.ne("exact").sum()),
                "excluded_external_rows": len(excluded), "post_filter_competition_overlap": 0,
                "source_sha256": hashlib.sha256(args.approved_external.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256(Path("vacv/interval_regression.py").read_bytes()).hexdigest(),
                "train_sha256": hashlib.sha256(Path("data/raw_v2/train_v2.csv").read_bytes()).hexdigest()}
    metrics = []

    def indices(train, profile):
        extra = ([] if profile == "competition-only" else
                 exact_indices if profile == "external-exact" else external_indices)
        return np.r_[train, extra].astype(int)

    def fit(train, profile, query_kernel):
        ix = indices(train, profile)
        model = IntervalKernelRegressor(alpha=1.).fit(kernel[np.ix_(ix, ix)], lower[ix], upper[ix], weights[ix])
        latent = model.predict(query_kernel[:, ix])
        with np.errstate(over="raise", invalid="raise"):
            prediction = np.maximum(0, np.expm1(latent))
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite unbounded regression prediction")
        return prediction

    with threadpool_limits(limits=2):
        for split in (21, 22, 23, "scaffold"):
            if split == "scaffold":
                groups = scaffold_folds(scaffolds_of(data.train), 5)
                iterator = [fold_indices(groups, f) for f in range(5)]
            else:
                iterator = KFold(5, shuffle=True, random_state=split).split(y)
            outputs = {name: np.full(len(y), np.nan) for name in PROFILES}
            folds = np.full(len(y), -1)
            for fold, (train, valid) in enumerate(iterator):
                for profile in PROFILES:
                    outputs[profile][valid] = fit(train, profile, kernel[valid])
                folds[valid] = fold
                print(f"split={split} fold={fold}", flush=True)
            for profile, prediction in outputs.items():
                error = (prediction-y)**2
                if not np.isfinite(error).all():
                    raise ValueError("Nonfinite full raw error")
                metrics.append({"split": split, "profile": profile, "mse": float(error.mean()),
                                "maximum_target_prediction": float(prediction[y.argmax()]),
                                "prediction_max": float(prediction.max())})
            pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds, **outputs}).to_csv(args.out/f"oof-{split}.csv", index=False)
            (args.out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            if split == 21:
                winner = min(metrics, key=lambda m: m["mse"])["profile"]
                manifest["frozen_winner"] = winner
            (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            print(metrics[-3:], flush=True)
        confirmations = [m["mse"] for m in metrics if m["profile"] == winner and m["split"] in (22, 23)]
        manifest["gate_pass"] = len(confirmations) == 2 and max(confirmations) <= 3_000_000
        if manifest["gate_pass"]:
            q = tanimoto(fingerprint(data.test), features)
            result = submission(data.test["index"], fit(np.arange(len(y)), winner, q))
            template = pd.read_csv("data/raw_v2/sample_submit.csv")
            result = result.set_index("index").loc[template["index"]].reset_index()
            path = args.out/"candidate.csv"
            result.to_csv(path, index=False)
            manifest["candidate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["status"] = "completed"
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
