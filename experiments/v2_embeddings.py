"""Bounded frozen-CheMeleon regression experiment on official v2 targets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from vacv.checkpoint import load_embeddings
from vacv.dataset import load, submission
from vacv.literature import presence, PRESENCE_FEATURES
from vacv.pipeline import ContextEncoder, scaffolds_of
from vacv.splits import scaffold_folds, fold_indices


def predict(train, y, query, embeddings):
    fields = ["SMILES", "Title", "strain", "Culture"]
    train, query = train[fields], query[fields]
    encoder = ContextEncoder().fit(train)
    pca = PCA(n_components=min(32, len(train)-1), random_state=0)
    ex = pca.fit_transform(np.stack([embeddings[s] for s in train.SMILES]))
    eq = pca.transform(np.stack([embeddings[s] for s in query.SMILES]))
    x = np.hstack([encoder.transform(train), ex])
    q = np.hstack([encoder.transform(query), eq])
    ext = presence()
    xp = train.merge(ext, on="SMILES", how="left", sort=False, validate="many_to_one")[PRESENCE_FEATURES].to_numpy(dtype=float)
    qp = query.merge(ext, on="SMILES", how="left", sort=False, validate="many_to_one")[PRESENCE_FEATURES].to_numpy(dtype=float)
    xp, qp = np.nan_to_num(xp, nan=-1), np.nan_to_num(qp, nan=-1)
    predictions = {}
    for name, log, external in (("chemeleon-et-raw", False, False),
                                ("chemeleon-et-log", True, False),
                                ("chemeleon-et-log-presence", True, True)):
        target = np.log1p(y) if log else y
        a, b = (np.hstack([x, xp]), np.hstack([q, qp])) if external else (x, q)
        pred = []
        for seed in (0, 1):
            model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                         max_features=.5, random_state=seed, n_jobs=1)
            pred.append(model.fit(a, target).predict(b))
        value = np.mean(pred, axis=0)
        predictions[name] = np.maximum(0, np.expm1(value) if log else value)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=10).fit(scaler.transform(x), np.log1p(y))
    predictions["chemeleon-ridge-log"] = np.maximum(0, np.expm1(np.minimum(model.predict(scaler.transform(q)), 20)))
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_v2"))
    parser.add_argument("--out", type=Path, default=Path("experiments/results/v2-embeddings"))
    parser.add_argument("--splits", nargs="+", default=["6", "7", "8", "scaffold"])
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output directory")
    data = load(args.data_dir, require_v2=True)
    embeddings, manifest = load_embeddings("data/raw/models/chemeleon_features.npz")
    args.out.mkdir(parents=True)
    metrics = []
    with threadpool_limits(limits=2):
        for split in args.splits:
            if split == "scaffold":
                folds = scaffold_folds(scaffolds_of(data.train), 5)
                iterator = [fold_indices(folds, k) for k in range(5)]
            else:
                iterator = KFold(5, shuffle=True, random_state=int(split)).split(data.y)
            outputs, ids = {}, np.full(len(data.y), -1)
            for k, (tr, va) in enumerate(iterator):
                values = predict(data.train.iloc[tr], data.y[tr], data.train.iloc[va], embeddings)
                for name, p in values.items():
                    outputs.setdefault(name, np.full(len(data.y), np.nan))[va] = p
                ids[va] = k
                print(f"{split} fold{k} done", flush=True)
            for name, p in outputs.items():
                if not np.isfinite(p).all():
                    raise ValueError("Nonfinite OOF")
                metrics.append({"split": split, "model": name, "mse": float(np.mean((p-data.y)**2))})
            pd.DataFrame({"index": np.arange(len(data.y)), "fold": ids, **outputs}).to_csv(args.out/f"oof-{split}.csv", index=False)
            (args.out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            print(metrics[-4:], flush=True)
        values = predict(data.train, data.y, data.test, embeddings)
    template = pd.read_csv(args.data_dir/"sample_submit.csv")
    metadata = {"target_version": "train_v2", "embedding_sha256": manifest["embedding_sha256"],
                "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "presence_sha256": hashlib.sha256(Path("data/processed/competition_presence.csv").read_bytes()).hexdigest(),
                "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (args.data_dir/"train_v2.csv", args.data_dir/"test.csv")}, "submissions": {}}
    for name, p in values.items():
        result = submission(data.test["index"].to_numpy(), p)
        if set(result["index"]) != set(template["index"]):
            raise ValueError("Template mismatch")
        result = result.set_index("index").loc[template["index"]].reset_index()
        path = args.out/f"submission-{name}.csv"
        result.to_csv(path, index=False)
        metadata["submissions"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(result)}
    (args.out/"submissions.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
