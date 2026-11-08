"""Actual SI regression after the official v2 target correction (sentinel -> zero).

No competition concentration columns, external lookups or test labels are read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from vacv.dataset import load, submission
from vacv.pipeline import ContextEncoder, structure_matrix, scaffolds_of
from vacv.kernel import KernelSelectivityModel
from vacv.splits import scaffold_folds, fold_indices
from vacv.regression_v2 import RegressionV2, PROFILES

FIELDS = ["index", "SMILES", "Title", "strain", "Culture"]


def predictions(train, y, query):
    train = train[[c for c in FIELDS if c in train]]
    query = query[[c for c in FIELDS if c in query]]
    encoder = ContextEncoder().fit(train)
    cx, cq = encoder.transform(train), encoder.transform(query)
    sx, sq = structure_matrix(train), structure_matrix(query)
    result = {"constant": np.full(len(query), y.mean())}
    with threadpool_limits(limits=2):
        for mode in ("raw", "log"):
            target = y if mode == "raw" else np.log1p(y)
            context, structure = [], []
            for seed in range(3):
                opts = dict(n_estimators=800, min_samples_leaf=2, max_features=.5, n_jobs=1, random_state=seed)
                context.append(ExtraTreesRegressor(**opts).fit(cx, target).predict(cq))
                structure.append(ExtraTreesRegressor(**opts).fit(sx, target).predict(sq))
            c, s = np.mean(context, 0), np.mean(structure, 0)
            for name, value in (("context", c), ("blend", .5*c+.5*s)):
                result[f"et-{name}-{mode}"] = np.maximum(0, value if mode == "raw" else np.expm1(value))
            # Raw/log kernel uses only structure and permitted context similarities.
            left, right = KernelSelectivityModel._inputs(train), KernelSelectivityModel._inputs(query)
            k = KernelSelectivityModel._kernel(left, left)
            q = KernelSelectivityModel._kernel(right, left)
            prior = target.mean()
            value = prior + q @ np.linalg.solve(k+np.eye(len(y)), target-prior)
            result[f"kernel-{mode}"] = np.maximum(0, value if mode == "raw" else np.expm1(np.minimum(value, 20)))
        model = CatBoostRegressor(iterations=500, depth=4, learning_rate=.03,
                                  l2_leaf_reg=5, loss_function="RMSE", verbose=False,
                                  thread_count=2, random_seed=0, allow_writing_files=False)
        x, q = np.column_stack([cx, sx]), np.column_stack([cq, sq])
        result["catboost-raw"] = np.maximum(0, model.fit(x, y).predict(q))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["6", "7", "8", "scaffold"])
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--profiles", nargs="+", choices=PROFILES)
    args = ap.parse_args()
    if args.out.exists(): ap.error("Choose a new output directory")
    args.out.mkdir(parents=True)
    data = load(args.data_dir, require_v2=True); y = data.y; metrics = []
    for split in args.splits:
        if split == "scaffold":
            folds = scaffold_folds(scaffolds_of(data.train), 5)
            iterator = [fold_indices(folds, k) for k in range(5)]
        else:
            iterator = KFold(5, shuffle=True, random_state=int(split)).split(y)
        outputs = {}; ids = np.full(len(y), -1)
        for k, (tr, va) in enumerate(iterator):
            values = ({name: RegressionV2(name).fit(data.train.iloc[tr], y[tr]).predict(data.train.iloc[va])
                       for name in args.profiles} if args.profiles else
                      predictions(data.train.iloc[tr], y[tr], data.train.iloc[va]))
            for name, p in values.items():
                outputs.setdefault(name, np.full(len(y), np.nan))[va] = p
            ids[va] = k
            print(f"{split} fold{k} done", flush=True)
        for name, p in outputs.items():
            assert np.isfinite(p).all()
            metrics.append({"split": split, "model": name, "mse": float(np.mean((p-y)**2))})
        pd.DataFrame({"index": np.arange(len(y)), "fold": ids, **outputs}).to_csv(args.out/f"oof-{split}.csv", index=False)
        (args.out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
        print(metrics[-len(outputs):], flush=True)
    if args.submit:
        values = ({name: RegressionV2(name).fit(data.train, y).predict(data.test)
                   for name in args.profiles} if args.profiles else predictions(data.train, y, data.test))
        template = pd.read_csv(args.data_dir/"sample_submit.csv")
        meta = {}
        for name, p in values.items():
            out = submission(data.test["index"].to_numpy(), p).set_index("index").loc[template["index"]].reset_index()
            assert list(out) == list(template)
            path = args.out/f"submission-{name}.csv"; out.to_csv(path, index=False)
            meta[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(out), "mean": float(p.mean())}
        train_name = "train_v2.csv" if (args.data_dir/"train_v2.csv").exists() else "train.csv"
        meta["data_sha256"] = {name: hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest()
                               for name in (train_name, "test.csv", "sample_submit.csv")}
        (args.out/"submissions.json").write_text(json.dumps(meta, indent=2)+"\n")


if __name__ == "__main__": main()
