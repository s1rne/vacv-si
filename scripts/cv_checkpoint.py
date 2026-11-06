"""Recompute complete checkpoint OOF predictions, including the unlabeled graph pool."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from vacv.checkpoint import BestCheckpointModel, load_embeddings
from vacv.dataset import load
from vacv.literature import attach
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/raw/models/chemeleon_features.npz"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/raw/models/tabicl-classifier-v2-20260212.ckpt"))
    parser.add_argument("--splits", nargs="+", default=["3", "4", "5", "scaffold"])
    parser.add_argument("--out", type=Path, default=Path("outputs/checkpoint-cv"))
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use an empty output directory")
    data = load()
    frame, test, y = attach(data.train), attach(data.test), data.y
    embeddings, _ = load_embeddings(args.features)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for split in args.splits:
        folds = (scaffold_folds(scaffolds_of(frame), 5) if split == "scaffold" else
                 [v for _, v in KFold(5, shuffle=True, random_state=int(split)).split(y)])
        prediction = np.full(len(y), np.nan)
        probability = np.full(len(y), np.nan)
        fold_id = np.full(len(y), -1)
        for fold in range(5):
            train, valid = fold_indices(folds, fold)
            model = BestCheckpointModel(embeddings, args.checkpoint).fit(frame.iloc[train], y[train])
            components = model.predict_components(frame.iloc[valid], unlabeled=test)
            prediction[valid], probability[valid] = components["prediction"], components["probability"]
            fold_id[valid] = fold
            print(f"split={split} fold={fold} complete", flush=True)
        if not np.isfinite(prediction).all():
            raise ValueError("Incomplete OOF predictions")
        row = {"split": split, "mse": float(np.mean((prediction - y) ** 2)), "model": "checkpoint"}
        results.append(row)
        pd.DataFrame({"index": np.arange(len(y)), "fold": fold_id, "probability": probability,
                      "prediction": prediction}).to_csv(args.out / f"oof-{split}.csv", index=False)
        (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
