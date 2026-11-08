"""Nested-CV graph residual correction, with frozen beta and one-hop ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from vacv.dataset import load, submission
from vacv.graph_correction import GraphResidualCorrector
from vacv.literature import attach
from vacv.pipeline import INACTIVE, scaffolds_of
from vacv.postprocess import conditional_lookup
from vacv.splits import fold_indices, scaffold_folds

CANDIDATES = [(mode, beta, False) for mode in ["onehop", "diffusion"] for beta in [.25, .5]]


def candidate_name(mode, beta, gated):
    return f"{'gated_' if gated else ''}{mode}_{beta:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/graph_screen"))
    parser.add_argument("--splits", nargs="+", default=["0", "1", "2", "scaffold"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--submit", choices=["onehop", "diffusion"])
    parser.add_argument("--beta", type=float, default=.25, help="Submission correction strength")
    parser.add_argument("--candidate", choices=["screen", "gated_diffusion_0.25"], default="screen")
    parser.add_argument("--with-soft", action="store_true", help="Compare conditional-mean literature substitution")
    args = parser.parse_args()
    if args.seeds < 1 or not np.isfinite(args.beta) or args.beta < 0:
        parser.error("Positive seeds and finite nonnegative beta required")
    if args.candidate != "screen" and args.submit and (args.submit != "diffusion" or args.beta != .25):
        parser.error("The frozen gated candidate uses diffusion and beta=0.25")
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        parser.error("Output directory is not empty; choose --out or explicitly pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)
    data = load()
    frame, test, y = attach(data.train), attach(data.test), data.y
    candidates = CANDIDATES if args.candidate == "screen" else [("diffusion", .25, True)]
    config = {"base_profile": "hybrid", "model_seed_count": args.seeds,
              "inner_splits": 4, "inner_seed": 2718, "outer_splits": [] if args.submit else args.splits,
              "unlabeled_test_rows": len(test), "candidates": candidates,
              "neighbours": 6, "cross_group": .05, "regularization": .03,
              "submit": args.submit, "submit_beta": args.beta if args.submit else None,
              "with_soft": args.with_soft}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.submit:
        estimator = GraphResidualCorrector(seeds=args.seeds).fit(frame, y)
        components = estimator.components(test)
        p, raw, prediction = estimator.predictions(test, components, args.beta, args.submit,
                                                   uncertainty_gate=args.candidate != "screen")
        if args.with_soft:
            prediction = conditional_lookup(raw, p, test)
        path = args.out / "submission.csv"
        submission(data.test["index"].to_numpy(), prediction).to_csv(path, index=False)
        metadata = {"rows": len(prediction), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "mode": args.submit, "beta": args.beta, "gated": args.candidate != "screen",
                    "with_soft": args.with_soft, "model_seed_count": args.seeds,
                    "inner_splits": 4, "inner_seed": 2718}
        (args.out / "submission_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {len(prediction)} predictions; not submitted to Kaggle", flush=True)
        return
    metrics = []
    for split in args.splits:
        folds = (scaffold_folds(scaffolds_of(frame)) if split == "scaffold" else
                 [valid for _, valid in KFold(5, shuffle=True, random_state=int(split)).split(y)])
        names = ["hybrid"] + [candidate_name(*candidate) for candidate in candidates]
        if args.with_soft:
            names += [f"{name}_soft" for name in names.copy()]
        predictions = {name: np.full(len(y), np.nan) for name in names}
        probabilities = {name: np.full(len(y), np.nan) for name in names}
        delta = {mode: np.full(len(y), np.nan) for mode in ["onehop", "diffusion"]}
        fold_ids = np.full(len(y), -1)
        for fold in range(len(folds)):
            train, valid = fold_indices(folds, fold)
            estimator = GraphResidualCorrector(seeds=args.seeds).fit(frame.iloc[train], y[train])
            components = estimator.components(frame.iloc[valid], unlabeled=test)
            predictions["hybrid"][valid] = components["final"]
            probabilities["hybrid"][valid] = components["probability"]
            if args.with_soft:
                predictions["hybrid_soft"][valid] = conditional_lookup(
                    components["raw"], components["probability"], frame.iloc[valid])
                probabilities["hybrid_soft"][valid] = components["probability"]
            fold_ids[valid] = fold
            for mode in delta:
                delta[mode][valid] = components["deltas"][mode]
            for mode, beta, gated in candidates:
                name = candidate_name(mode, beta, gated)
                probability, raw, prediction = estimator.predictions(frame.iloc[valid], components, beta, mode,
                                                                      uncertainty_gate=gated)
                probabilities[name][valid], predictions[name][valid] = probability, prediction
                if args.with_soft:
                    predictions[f"{name}_soft"][valid] = conditional_lookup(raw, probability, frame.iloc[valid])
                    probabilities[f"{name}_soft"][valid] = probability
            print(f"{split} fold {fold + 1}/5 complete", flush=True)
        rows = {"row_index": np.arange(len(y)), "fold": fold_ids}
        rows.update({f"prediction_{name}": p for name, p in predictions.items()})
        rows.update({f"probability_{name}": p for name, p in probabilities.items()})
        rows.update({f"delta_{mode}": d for mode, d in delta.items()})
        pd.DataFrame(rows).to_csv(args.out / f"oof_{split}.csv", index=False)
        for name in names:
            if not np.isfinite(predictions[name]).all():
                raise ValueError("Incomplete OOF predictions")
            mse = float(np.mean((predictions[name] - y) ** 2))
            brier = float(np.mean((probabilities[name] - (y == INACTIVE)) ** 2))
            metrics.append({"split": split, "model": name, "mse": mse, "brier": brier})
            print(f"{split:8s} {name:20s} MSE {mse:,.3f} Brier {brier:.6f}", flush=True)
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
