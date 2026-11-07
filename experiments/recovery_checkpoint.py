"""Compare original model weights and the checkpoint on target-independent inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from vacv.checkpoint import BestCheckpointModel, load_embeddings
from vacv.clean_features import attach_clean
from vacv.dataset import load
from vacv.pipeline import SelectivityModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", type=int, default=[3, 6, 7, 8])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output must be new")
    args.out.mkdir(parents=True)
    data = load()
    train, test = attach_clean(data.train), attach_clean(data.test)
    embeddings, _ = load_embeddings("data/raw/models/chemeleon_features.npz")
    checkpoint = "data/raw/models/tabicl-classifier-v2-20260212.ckpt"
    metrics = []
    for seed in args.splits:
        predictions = {name: np.full(len(train), np.nan) for name in ("original_weights", "checkpoint")}
        folds = np.full(len(train), -1)
        for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=seed).split(train)):
            with threadpool_limits(limits=4):
                baseline = [SelectivityModel(seed=s, weights={"structure": 0., "context": .3, "grouped": .7})
                            .fit(train.iloc[tr], data.y[tr]).predict(train.iloc[va]) for s in range(7)]
                predictions["original_weights"][va] = np.mean(baseline, axis=0)
                model = BestCheckpointModel(embeddings, checkpoint).fit(train.iloc[tr], data.y[tr])
                predictions["checkpoint"][va] = model.predict(train.iloc[va], unlabeled=test)
            folds[va] = fold
            print(f"split={seed} fold={fold} complete", flush=True)
        for name, pred in predictions.items():
            if not np.isfinite(pred).all():
                raise ValueError("Incomplete OOF")
            metrics.append({"split": seed, "model": name, "mse": float(np.mean((pred-data.y)**2))})
        pd.DataFrame({"index": np.arange(len(train)), "fold": folds, **predictions}).to_csv(args.out/f"oof-{seed}.csv", index=False)
        (args.out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
        print(metrics[-2:], flush=True)


if __name__ == "__main__":
    main()
