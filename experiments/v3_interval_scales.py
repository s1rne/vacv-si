"""Frozen raw/sqrt scale ablation for audited exact and interval SI evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v3_intervals import fingerprint, read_snapshot, tanimoto
from vacv.dataset import load, submission
from vacv.interval_regression import IntervalKernelRegressor
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

SOURCE_HASH = "9ead9fc63139b88352584a39673f3458beaf1abb8c3ddd776c0de7d1729a2e23"
SCALES = ("raw", "sqrt")
MODES = ("competition-only", "external-exact", "external-intervals")
PROFILES = tuple(f"{scale}/{mode}" for scale in SCALES for mode in MODES)


def transform(value, scale):
    value = np.asarray(value, dtype=float)
    if np.isnan(value).any() or (value < 0).any():
        raise ValueError("Nonnegative targets/bounds without NaN required")
    if scale == "raw":
        return value / 1000.
    if scale == "sqrt":
        return np.sqrt(value)
    raise ValueError("Unknown target scale")


def inverse(value, scale):
    value = np.asarray(value, dtype=float)
    if np.isnan(value).any():
        raise ValueError("No NaN predictions")
    value = np.maximum(value, 0.)
    if scale == "raw":
        return value * 1000.
    if scale == "sqrt":
        return value**2
    raise ValueError("Unknown target scale")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-external", type=Path, default=Path("experiments/results/v3-source-pairs/eligible.csv"))
    parser.add_argument("--out", type=Path, default=Path("experiments/results/v3-interval-scales"))
    args = parser.parse_args()
    if hashlib.sha256(args.approved_external.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("Source changed; this experiment requires the audited frozen snapshot")
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
    ext_ix = np.arange(len(y), len(features))
    exact_ix = ext_ix[external.kind.eq("exact").to_numpy()]
    weights = np.r_[np.ones(len(y)), np.full(len(external), .25)]
    lowers = {s: transform(np.r_[y, external.lower_si.to_numpy(float)], s) for s in SCALES}
    uppers = {s: transform(np.r_[y, external.upper_si.to_numpy(float)], s) for s in SCALES}
    manifest = {"profiles": PROFILES, "alpha": 1., "external_weight": .25,
                "discovery": 27, "confirmation": [28, 29], "target_version": "train_v2",
                "train_rows": len(y), "train_maximum": float(y.max()),
                "external_rows": len(external), "external_exact": int(external.kind.eq("exact").sum()),
                "external_bounds": int(external.kind.ne("exact").sum()),
                "excluded_external_rows": len(excluded), "post_filter_competition_overlap": 0,
                "source_sha256": SOURCE_HASH, "source": str(args.approved_external),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256(Path("vacv/interval_regression.py").read_bytes()).hexdigest(),
                "helper_sha256": hashlib.sha256(Path("experiments/v3_intervals.py").read_bytes()).hexdigest(),
                "train_sha256": hashlib.sha256(Path("data/raw_v2/train_v2.csv").read_bytes()).hexdigest()}
    metrics = []

    def fit(train, profile, query_kernel):
        scale, mode = profile.split("/")
        extra = [] if mode == "competition-only" else exact_ix if mode == "external-exact" else ext_ix
        ix = np.r_[train, extra].astype(int)
        model = IntervalKernelRegressor(alpha=1.).fit(kernel[np.ix_(ix, ix)], lowers[scale][ix], uppers[scale][ix], weights[ix])
        with np.errstate(over="raise", invalid="raise"):
            prediction = inverse(model.predict(query_kernel[:, ix]), scale)
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite unbounded prediction")
        return prediction

    try:
        with threadpool_limits(limits=2):
            for split in (27, 28, 29, "scaffold"):
                if split == "scaffold":
                    groups = scaffold_folds(scaffolds_of(data.train), 5)
                    iterator = [fold_indices(groups, fold) for fold in range(5)]
                else:
                    iterator = KFold(5, shuffle=True, random_state=split).split(y)
                profiles = PROFILES if split == 27 else tuple(dict.fromkeys((winner, baseline)))
                outputs = {p: np.full(len(y), np.nan) for p in profiles}
                folds = np.full(len(y), -1)
                for fold, (train, valid) in enumerate(iterator):
                    for profile in profiles:
                        outputs[profile][valid] = fit(train, profile, kernel[valid])
                    folds[valid] = fold
                    print(f"split={split} fold={fold}", flush=True)
                for profile, prediction in outputs.items():
                    with np.errstate(over="raise", invalid="raise"):
                        errors = (prediction-y)**2
                    if not np.isfinite(errors).all():
                        raise ValueError("Nonfinite full raw error")
                    metrics.append({"split": split, "profile": profile, "mse": float(errors.mean()),
                                    "maximum_target_prediction": float(prediction[y.argmax()]),
                                    "prediction_max": float(prediction.max())})
                pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds, **outputs}).to_csv(args.out/f"oof-{split}.csv", index=False)
                (args.out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
                if split == 27:
                    winner = min(metrics, key=lambda item: item["mse"])["profile"]
                    baseline = winner.split("/")[0]+"/competition-only"
                    manifest.update(frozen_winner=winner, matching_baseline=baseline)
                (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
                print(metrics[-len(profiles):], flush=True)
            confirmations = [m["mse"] for m in metrics if m["profile"] == winner and m["split"] in (28, 29)]
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
    except (ValueError, RuntimeError, FloatingPointError) as error:
        manifest.update(status="failed", gate_pass=False, failure=str(error))
        raise
    finally:
        (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
