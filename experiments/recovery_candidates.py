"""Bounded clean-input active-value / probability-temperature ablation.

No competition concentration columns, supervised lookup or test labels are read.
All active-value estimates use only the current outer training fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from vacv.dataset import load
from vacv.inference import predict_ensemble
from vacv.literature import presence
from vacv.pipeline import INACTIVE, structure_matrix, scaffolds_of
from vacv.splits import scaffold_folds, fold_indices


def clean_frame(frame):
    out = frame.merge(presence(), on="SMILES", how="left", validate="many_to_one")
    out["matched"] = 0.0
    out["literature_si"] = np.nan
    out["literature_quality"] = None
    return out


def active_knn(train, y, query):
    """Smooth active SI using molecular similarity, never concentration proxies."""
    active = y != INACTIVE
    if not active.any():
        return np.zeros(len(query))
    a = (structure_matrix(query)[:, :2048] > 0).astype(float)
    b = (structure_matrix(train.iloc[np.flatnonzero(active)])[:, :2048] > 0).astype(float)
    dot = a @ b.T
    denom = a.sum(1)[:, None] + b.sum(1)[None, :] - dot
    similarity = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    weights = similarity ** 4
    prior = y[active].mean()
    return (weights @ y[active] + 3 * prior) / (weights.sum(1) + 3)


def temperature(p, exponent):
    left, right = p ** exponent, (1 - p) ** exponent
    return left / (left + right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["3"])
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory must not already exist")
    args.out.mkdir(parents=True)
    data = load()
    frame, y = clean_frame(data.train), data.y
    results = []
    for split in args.splits:
        split = split if split == "scaffold" else int(split)
        if split == "scaffold":
            scaffold = scaffold_folds(scaffolds_of(frame), 5)
            folds = [fold_indices(scaffold, fold) for fold in range(5)]
        else:
            folds = KFold(5, shuffle=True, random_state=split).split(y)
        rows = []
        for fold, (train, valid) in enumerate(folds):
            with threadpool_limits(limits=2):
                p, raw, _ = predict_ensemble(frame.iloc[train], y[train], frame.iloc[valid], args.seeds, "hybrid")
                active = active_knn(frame.iloc[train], y[train], frame.iloc[valid])
            mean = y[train][y[train] != INACTIVE].mean()
            row = pd.DataFrame({"index": valid, "fold": fold, "target": y[valid],
                                "probability": p, "hybrid": raw,
                                "active_knn": p * INACTIVE + (1 - p) * active})
            for exponent in (0.8, 1.2):
                q = temperature(p, exponent)
                row[f"temperature_{exponent}"] = q * INACTIVE + (1 - q) * mean
            rows.append(row)
            print(f"split={split} fold={fold} done", flush=True)
        oof = pd.concat(rows).sort_values("index")
        oof.to_csv(args.out / f"oof-{split}.csv", index=False)
        for name in ("hybrid", "active_knn", "temperature_0.8", "temperature_1.2"):
            results.append({"split": split, "model": name,
                            "mse": float(np.mean((oof[name] - oof.target) ** 2))})
        (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results[-4:]), flush=True)


if __name__ == "__main__":
    main()
