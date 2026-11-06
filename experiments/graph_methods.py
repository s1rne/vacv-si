"""Evaluate transductive graph candidates with no validation/test label access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.literature import apply, attach
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices
from vacv.transductive import HarmonicSelectivityModel

CONFIGS = {
    "graph": {},
    "graph_local": {"cross_group": 0.0, "regularization": 0.01},
    "graph_evidence": {"evidence_strength": 0.2},
    "graph_smooth": {"neighbours": 12, "regularization": 0.1},
    "source_logistic": {"C": 3.0},
    "source_logistic10": {"C": 10.0},
    "tabicl": {"n_estimators": 2},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/graphs"))
    parser.add_argument("--profiles", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a new output directory to preserve previous runs")
    args.out.mkdir(parents=True, exist_ok=True)
    data = load()
    frame, test, y = attach(data.train), attach(data.test), data.y
    folds = {f"random-{seed}": [v for _, v in KFold(5, shuffle=True, random_state=seed).split(y)]
             for seed in range(args.start_seed, args.start_seed + args.repeats)}
    folds["scaffold"] = scaffold_folds(scaffolds_of(frame), 5)
    results = []
    for split, partitions in folds.items():
        for name in args.profiles:
            prediction = np.empty(len(y))
            probability = np.empty(len(y))
            fold_id = np.empty(len(y), dtype=int)
            for fold in range(len(partitions)):
                train, valid = fold_indices(partitions, fold)
                if name == "tabicl":
                    from vacv.foundation import FoundationSelectivityModel
                    model = FoundationSelectivityModel(**CONFIGS[name])
                elif name.startswith("source"):
                    from vacv.source_kernel import SourceKernelClassifier
                    model = SourceKernelClassifier(**CONFIGS[name])
                else:
                    model = HarmonicSelectivityModel(**CONFIGS[name])
                p, raw = model.fit_predict(
                    frame.iloc[train], y[train], frame.iloc[valid], test)
                prediction[valid] = apply(raw, frame.iloc[valid])
                probability[valid] = p
                fold_id[valid] = fold
            mse = float(np.mean((prediction - y) ** 2))
            results.append({"split": split, "profile": name, "mse": mse,
                            "parameters": CONFIGS[name]})
            pd.DataFrame({"index": np.arange(len(y)), "fold": fold_id, "target": y,
                          "probability": probability, "prediction": prediction}).to_csv(
                              args.out / f"{split}-{name}.csv", index=False)
            (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"{split:12} {name:16} MSE={mse:,.0f}", flush=True)


if __name__ == "__main__":
    main()
