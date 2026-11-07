"""Compare the actual submission pipeline on repeated held-out folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.inference import PROFILES, predict_ensemble
from vacv.literature import attach, fold_table, to_zero_inactive
from vacv.pipeline import INACTIVE, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def evaluate(raw_frame, y, folds, seeds=7, profile="baseline"):
    probability, raw, prediction, zero = (np.full(len(y), np.nan) for _ in range(4))
    fold_ids = np.full(len(y), -1)
    for fold in range(len(folds)):
        train, valid = fold_indices(folds, fold)
        if not len(train) or not len(valid):
            raise ValueError("Empty training or validation fold")
        # Правила подстановки пересобираются по обучающей части фолда: иначе строки
        # контрольной части участвуют в решении, подставлять ли их собственный бакет.
        frame = attach(raw_frame, fold_table(train))
        probability[valid], raw[valid], prediction[valid] = predict_ensemble(
            frame.iloc[train], y[train], frame.iloc[valid], seeds, profile)
        zero[valid] = to_zero_inactive(prediction[valid], probability[valid], frame.iloc[valid])
        fold_ids[valid] = fold
    if not np.isfinite(prediction).all() or (fold_ids < 0).any():
        raise ValueError("Incomplete OOF predictions")
    labels = y == INACTIVE
    metrics = {
        "mse": float(np.mean((prediction - y) ** 2)),
        # метки train_v2: неактивное соединение имеет SI = 0, это и считает Kaggle
        "mse_zero_inactive": float(np.mean((zero - np.where(y == INACTIVE, 0.0, y)) ** 2)),
        "model_mse": float(np.mean((raw - y) ** 2)),
        "auc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else None,
        "brier": float(np.mean((probability - labels) ** 2)),
    }
    row_ids = raw_frame["index"].to_numpy() if "index" in raw_frame else np.arange(len(raw_frame))
    oof = pd.DataFrame({"index": row_ids, "fold": fold_ids,
                        "target": y, "probability": probability,
                        "model_prediction": raw, "prediction": prediction})
    return metrics, oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=["baseline", "hybrid"])
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=0,
                        help="First random split seed; use fresh seeds for confirmation")
    parser.add_argument("--split", choices=["random", "scaffold", "both"], default="both")
    parser.add_argument("--out", type=Path, default=Path("outputs/cv"))
    args = parser.parse_args()
    if args.repeats < 1 or args.seeds < 1:
        parser.error("seeds and repeats must be positive")
    data = load()
    frame, y = data.train, data.y
    splits = {}
    if args.split in {"random", "both"}:
        for repeat in range(args.start_seed, args.start_seed + args.repeats):
            splits[f"random-{repeat}"] = [v for _, v in KFold(5, shuffle=True, random_state=repeat).split(y)]
    if args.split in {"scaffold", "both"}:
        splits["scaffold"] = scaffold_folds(scaffolds_of(frame), n_splits=5)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, folds in splits.items():
        for profile in args.profiles:
            metrics, oof = evaluate(frame, y, folds, args.seeds, profile)
            results.append({"split": name, "profile": profile, "seeds": args.seeds, **metrics})
            oof.to_csv(args.out / f"{name}-{profile}.csv", index=False)
            (args.out / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
            print(f"{name:12} {profile:12} MSE={metrics['mse']:,.0f} "
                  f"model={metrics['model_mse']:,.0f} AUC={metrics['auc']}", flush=True)


if __name__ == "__main__":
    main()
