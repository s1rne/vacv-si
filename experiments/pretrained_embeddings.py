"""Leakage-safe screening of frozen CheMeleon heads against the shared hybrid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from vacv.dataset import load
from vacv.inference import predict_ensemble
from vacv.kernel import KernelSelectivityModel
from vacv.literature import apply, attach, PRESENCE_FEATURES
from vacv.pipeline import INACTIVE, SENTINEL, scaffolds_of, title_matrix
from vacv.postprocess import conditional_lookup
from vacv.splits import fold_indices, scaffold_folds

PROFILES = ("che-ridge", "che-additive", "che-svr", "che-tabicl")


class FoldEmbedding:
    """PCA and kernel bandwidth are fitted exclusively on the training fold."""

    def __init__(self, components=64):
        self.components = components

    def fit(self, train):
        self.pca_ = PCA(n_components=min(self.components, *train.shape), svd_solver="full")
        reduced = self.pca_.fit_transform(train)
        self.gamma_ = 1.0 / max(float(np.var(reduced) * reduced.shape[1]), 1e-12)
        return self

    def transform(self, x):
        return self.pca_.transform(x)


def categorical_kernel(left, right):
    a = left[["strain", "Culture"]].to_numpy()
    b = right[["strain", "Culture"]].to_numpy()
    group = (a[:, None] == b[None, :]).all(axis=2)
    evidence = left.matched.to_numpy()[:, None] == right.matched.to_numpy()[None, :]
    return 2 * group + group * evidence


def tabular(frame, x):
    result = frame[["strain", "Culture"]].reset_index(drop=True).astype(str)
    result["group"] = result.strain + "|" + result.Culture
    for i, column in enumerate(title_matrix(frame.Title).T):
        result[f"title_{i}"] = column
    for name in ["matched", *PRESENCE_FEATURES]:
        result[name] = frame[name].to_numpy(float)
    return pd.concat([result, pd.DataFrame(x, columns=[f"che_{i}" for i in range(x.shape[1])])], axis=1)


def predict_head(train, y, valid, embedding_train, embedding_valid, profile, checkpoint):
    labels = (y == INACTIVE).astype(float)
    if len(np.unique(labels)) == 1:
        return np.full(len(valid), labels[0])
    reducer = FoldEmbedding(32 if profile == "che-tabicl" else 64).fit(embedding_train)
    a, b = reducer.transform(embedding_train), reducer.transform(embedding_valid)
    if profile == "che-tabicl":
        import torch
        from tabicl import TabICLClassifier
        torch.set_num_threads(4)
        model = TabICLClassifier(n_estimators=2, batch_size=1, device="cpu", use_amp=False,
                                n_jobs=4, model_path=str(checkpoint), allow_auto_download=False,
                                random_state=0)
        model.fit(tabular(train, a), labels.astype(int))
        return model.predict_proba(tabular(valid, b))[:, list(model.classes_).index(1)]
    aa = rbf_kernel(a, gamma=reducer.gamma_) + categorical_kernel(train, train)
    ba = rbf_kernel(b, a, gamma=reducer.gamma_) + categorical_kernel(valid, train)
    if profile == "che-additive":
        ki = KernelSelectivityModel._inputs
        kk = KernelSelectivityModel._kernel
        aa += kk(ki(train), ki(train)) - categorical_kernel(train, train)
        ba += kk(ki(valid), ki(train)) - categorical_kernel(valid, train)
    if profile == "che-svr":
        prediction = SVR(kernel="precomputed", C=1.0, epsilon=0.05).fit(aa, labels).predict(ba)
    else:
        prior = labels.mean()
        prediction = prior + ba @ np.linalg.solve(aa + 0.3 * np.eye(len(y)), labels - prior)
    return np.clip(prediction, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/raw/models/chemeleon_features.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/raw/models/tabicl-classifier-v2-20260212.ckpt"))
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--seeds", type=int, default=1, help="Model seeds for the hybrid comparator only")
    parser.add_argument("--soft", action="store_true", help="Also report conditional-mean literature substitution")
    parser.add_argument("--splits", nargs="+", default=["random-0", "random-1", "random-2", "scaffold"])
    parser.add_argument("--out", type=Path, default=Path("outputs/chemeleon-screen"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds must be positive")
    if "che-tabicl" in args.profiles:
        expected = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != expected:
            raise ValueError("TabICL checkpoint checksum mismatch")
    data = load()
    frame, y = attach(data.train), data.y
    manifest = json.loads(args.features.with_suffix(".json").read_text())
    if hashlib.sha256(args.features.read_bytes()).hexdigest() != manifest["embedding_sha256"]:
        raise ValueError("Embedding cache checksum mismatch")
    with np.load(args.features, allow_pickle=False) as archive:
        by_smiles = dict(zip(archive["smiles"], archive["embeddings"], strict=True))
    embeddings = np.stack([by_smiles[s] for s in frame.SMILES])
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite embedding cache")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with threadpool_limits(limits=4):
        for split in args.splits:
            folds = (scaffold_folds(scaffolds_of(frame), 5) if split == "scaffold" else
                     [v for _, v in KFold(5, shuffle=True, random_state=int(split.split("-")[1])).split(y)])
            names = ["hybrid", *args.profiles, *[f"hybrid+{p}" for p in args.profiles]]
            values = {name: np.full((len(y), 3), np.nan) for name in names}
            fold_ids = np.full(len(y), -1)
            for fold in range(5):
                train, valid = fold_indices(folds, fold)
                tr, va = frame.iloc[train], frame.iloc[valid]
                base = np.column_stack(predict_ensemble(tr, y[train], va, args.seeds, "hybrid"))
                values["hybrid"][valid] = base
                active = y[train][y[train] != INACTIVE]
                active_mean = active.mean() if len(active) else 0.0
                for profile in args.profiles:
                    probability = predict_head(tr, y[train], va, embeddings[train], embeddings[valid], profile, args.checkpoint)
                    raw = probability * SENTINEL + (1 - probability) * active_mean
                    head = np.column_stack([probability, raw, apply(raw, va)])
                    values[profile][valid] = head
                    values[f"hybrid+{profile}"][valid] = 0.75 * base + 0.25 * head
                    print(f"{split} fold {fold} {profile} complete", flush=True)
                fold_ids[valid] = fold
            if args.soft:
                for name, value in list(values.items()):
                    soft_value = value.copy()
                    soft_value[:, 2] = conditional_lookup(value[:, 1], value[:, 0], frame)
                    values[f"{name}-soft"] = soft_value
            for name, value in values.items():
                if not np.isfinite(value).all():
                    raise ValueError("Incomplete OOF")
                probability, raw, prediction = value.T
                row_ids = frame["index"].to_numpy() if "index" in frame else np.arange(len(frame))
                oof = pd.DataFrame({"index": row_ids, "fold": fold_ids, "target": y,
                                    "probability": probability, "model_prediction": raw, "prediction": prediction})
                path = args.out / f"{split}-{name}.csv"
                oof.to_csv(path, index=False)
                row = {"split": split, "profile": name, "hybrid_seeds": args.seeds,
                       "mse": float(np.mean((prediction-y)**2)),
                       "model_mse": float(np.mean((raw-y)**2)), "auc": float(roc_auc_score(y == INACTIVE, probability)),
                       "brier": float(np.mean((probability-(y == INACTIVE))**2)),
                       "oof_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                results.append(row)
                print(f"{split} {name} MSE {row['mse']:,.0f}", flush=True)
            (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
