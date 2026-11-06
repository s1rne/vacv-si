"""Post-hoc exploratory confidence gating of previously computed nested OOF residuals.

No new model is fitted. Every corrected probability uses its existing outer-fold
delta, and conversion to SI uses only that outer fold's training active mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.graph_correction import correct_probability
from vacv.literature import apply, attach
from vacv.pipeline import INACTIVE, SENTINEL, scaffolds_of
from vacv.splits import scaffold_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("results/graph_screen"))
    parser.add_argument("--out", type=Path, default=Path("results/graph_gated"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        parser.error("Output directory is not empty; choose --out or explicitly pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)
    data = load()
    frame, y = attach(data.train), data.y
    config = json.loads((args.source / "config.json").read_text())
    metrics = []
    for split in config["outer_splits"]:
        original = pd.read_csv(args.source / f"oof_{split}.csv")
        if not np.array_equal(original.row_index, np.arange(len(y))):
            raise ValueError("Source OOF row order mismatch")
        folds = (scaffold_folds(scaffolds_of(frame)) if split == "scaffold" else
                 [valid for _, valid in KFold(5, shuffle=True, random_state=int(split)).split(y)])
        expected_fold = np.full(len(y), -1)
        for fold, valid in enumerate(folds):
            expected_fold[valid] = fold
        if not np.array_equal(original.fold, expected_fold):
            raise ValueError("Source OOF folds mismatch")
        rows = {"row_index": np.arange(len(y)), "fold": expected_fold}
        for beta in [.25, .5]:
            p = correct_probability(original.probability_hybrid.to_numpy(),
                                    original.delta_diffusion.to_numpy(), beta, uncertainty_gate=True)
            raw = np.full(len(y), np.nan)
            for fold in range(len(folds)):
                valid = expected_fold == fold
                active = y[(~valid) & (y != INACTIVE)]
                mean = active.mean() if len(active) else 0.
                raw[valid] = p[valid] * SENTINEL + (1 - p[valid]) * mean
            prediction = apply(raw, frame)
            name = f"gated_diffusion_{beta:g}"
            mse = float(np.mean((prediction - y) ** 2))
            rows[f"prediction_{name}"] = prediction
            rows[f"probability_{name}"] = p
            metrics.append({"split": split, "model": name, "mse": mse,
                            "brier": float(np.mean((p - (y == INACTIVE)) ** 2)),
                            "status": "post-hoc exploratory, not confirmation"})
            print(f"{split:8s} {name:24s} MSE {mse:,.3f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out / f"oof_{split}.csv", index=False)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
