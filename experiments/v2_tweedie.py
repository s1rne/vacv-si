"""Frozen two-candidate Tweedie experiment; full official target and fold-local encoding."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from vacv.dataset import load, submission
from vacv.pipeline import ContextEncoder, structure_matrix, scaffolds_of
from vacv.regression_v2 import allowed
from vacv.splits import scaffold_folds, fold_indices


def predict(train, y, query, power):
    train, query = allowed(train), allowed(query)
    encoder = ContextEncoder().fit(train)
    x = np.column_stack([encoder.transform(train), structure_matrix(train)])
    q = np.column_stack([encoder.transform(query), structure_matrix(query)])
    # Fixed units conversion preserves all targets and does not inspect validation.
    model = CatBoostRegressor(iterations=600, depth=4, learning_rate=.025,
        l2_leaf_reg=10, loss_function=f"Tweedie:variance_power={power}",
        random_seed=0, thread_count=2, verbose=False, allow_writing_files=False)
    model.fit(x, np.asarray(y)/1000)
    value = model.predict(q, prediction_type="Exponent")*1000
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Invalid prediction")
    return value


def main():
    out = Path("experiments/results/v2-tweedie")
    out.mkdir(parents=True, exist_ok=False)
    data = load(Path("data/raw_v2"), require_v2=True)
    metrics = []
    candidates = {"tweedie-1.5": 1.5, "tweedie-1.8": 1.8}
    with threadpool_limits(limits=2):
        for split in (9, 10, 11, "scaffold"):
            iterator = (KFold(5, shuffle=True, random_state=split).split(data.y)
                        if split != "scaffold" else
                        [fold_indices(scaffold_folds(scaffolds_of(data.train), 5), k) for k in range(5)])
            predictions = {name: np.full(len(data.y), np.nan) for name in candidates}
            folds = np.full(len(data.y), -1)
            for fold, (tr, va) in enumerate(iterator):
                for name, power in candidates.items():
                    predictions[name][va] = predict(data.train.iloc[tr], data.y[tr], data.train.iloc[va], power)
                folds[va] = fold
                print(f"split={split} fold={fold}", flush=True)
            for name, p in predictions.items():
                metrics.append({"split": split, "model": name, "mse": float(np.mean((p-data.y)**2))})
            pd.DataFrame({"index": np.arange(len(data.y)), "fold": folds, **predictions}).to_csv(out/f"oof-{split}.csv", index=False)
            (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            print(metrics[-len(candidates):], flush=True)
            if split == 9:
                winner = min(metrics, key=lambda item: item["mse"])["model"]
                candidates = {winner: candidates[winner]}
        template = pd.read_csv("data/raw_v2/sample_submit.csv")
        result = submission(data.test["index"], predict(data.train, data.y, data.test, candidates[winner]))
        if set(result["index"]) != set(template["index"]):
            raise ValueError("Template mismatch")
        result = result.set_index("index").loc[template["index"]].reset_index()
        path = out/f"submission-{winner}.csv"
        result.to_csv(path, index=False)
    manifest = {"winner": winner, "selection_split": 9, "confirmation_splits": [10, 11, "scaffold"],
                "target_version": "train_v2", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "train_sha256": hashlib.sha256(Path("data/raw_v2/train_v2.csv").read_bytes()).hexdigest()}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
