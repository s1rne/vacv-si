"""Recompute complete checkpoint OOF predictions, including the unlabeled graph pool."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from vacv.checkpoint import BestCheckpointModel, load_embeddings
from vacv.clean_features import attach_clean
from vacv.dataset import load
from vacv.target_version import add_legacy_target_flag, check_legacy_target
from vacv.literature import attach, fold_table
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/raw/models/chemeleon_features.npz"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/raw/models/tabicl-classifier-v2-20260212.ckpt"))
    parser.add_argument("--splits", nargs="+", default=["3", "4", "5", "scaffold"])
    parser.add_argument("--out", type=Path, default=Path("outputs/checkpoint-cv"))
    parser.add_argument("--active-value", action="store_true",
                        help="Replace the global active mean with the per-row estimate")
    parser.add_argument("--lookup-policy", choices=["clean", "recovered"], default="clean",
                        help="Clean uses no label-fitted lookup; recovered requires the source corpus")
    add_legacy_target_flag(parser)
    args = parser.parse_args()
    check_legacy_target(parser, args)
    if args.active_value:
        parser.error("Legacy --active-value requires excluded concentration features and is disabled")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use an empty output directory")
    data = load()
    y = data.y
    embeddings, _ = load_embeddings(args.features)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for split in args.splits:
        folds = (scaffold_folds(scaffolds_of(data.train), 5) if split == "scaffold" else
                 [v for _, v in KFold(5, shuffle=True, random_state=int(split)).split(y)])
        prediction = np.full(len(y), np.nan)
        probability = np.full(len(y), np.nan)
        fold_id = np.full(len(y), -1)
        for fold in range(5):
            train, valid = fold_indices(folds, fold)
            # Правила подстановки пересобираются по обучающей части фолда, иначе
            # контрольные строки голосуют за решение о собственном бакете.
            if args.lookup_policy == "clean":
                frame, test = attach_clean(data.train), attach_clean(data.test)
            else:
                table = fold_table(train)
                frame, test = attach(data.train, table), attach(data.test, table)
            model = BestCheckpointModel(embeddings, args.checkpoint,
                                        active_value=args.active_value).fit(frame.iloc[train], y[train])
            components = model.predict_components(frame.iloc[valid], unlabeled=test)
            prediction[valid], probability[valid] = components["prediction"], components["probability"]
            fold_id[valid] = fold
            print(f"split={split} fold={fold} complete", flush=True)
        if not np.isfinite(prediction).all():
            raise ValueError("Incomplete OOF predictions")
        row = {"split": split, "mse": float(np.mean((prediction - y) ** 2)),
               "lookup_policy": args.lookup_policy,
               "model": "checkpoint-av" if args.active_value else "checkpoint"}
        results.append(row)
        pd.DataFrame({"index": np.arange(len(y)), "fold": fold_id, "probability": probability,
                      "prediction": prediction}).to_csv(args.out / f"oof-{split}.csv", index=False)
        (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
