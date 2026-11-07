"""Bounded TabICL v2 raw/log comparison, using fold-local context and PCA."""
import json
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from tabicl import TabICLRegressor
from huggingface_hub import hf_hub_download
from threadpoolctl import threadpool_limits

from vacv.checkpoint import load_embeddings
from vacv.dataset import load, submission
from vacv.pipeline import ContextEncoder, scaffolds_of
from vacv.splits import scaffold_folds, fold_indices


def features(train, query, embeddings):
    fields = ["SMILES", "Title", "strain", "Culture"]
    train, query = train[fields], query[fields]
    context = ContextEncoder().fit(train)
    pca = PCA(n_components=min(32, len(train)-1), random_state=0)
    a = pca.fit_transform(np.stack([embeddings[s] for s in train.SMILES]))
    b = pca.transform(np.stack([embeddings[s] for s in query.SMILES]))
    return np.hstack([context.transform(train), a]), np.hstack([context.transform(query), b])


def predict(a, y, b, log):
    model = TabICLRegressor(n_estimators=4, batch_size=1, device="cpu", n_jobs=2,
                           random_state=0, use_amp=False, use_fa3=False)
    model.fit(a, np.log1p(y) if log else y)
    p = model.predict(b)
    p = np.maximum(0, np.expm1(np.minimum(p, 20)) if log else p)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite TabICL prediction")
    return p


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    out = Path("experiments/results/v2-tabicl")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    embeddings, emb_manifest = load_embeddings("data/raw/models/chemeleon_features.npz")
    manifest = {"checkpoint": "tabicl-regressor-v2-20260212.ckpt", "device": "cpu", "n_estimators": 4,
                "discovery": 9, "confirmation": [10, 11], "embedding_sha256": emb_manifest["embedding_sha256"],
                "train_sha256": hashlib.sha256(Path("data/raw_v2/train_v2.csv").read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "features": "fold-local ContextEncoder and PCA32 of frozen CheMeleon", "target_version": "train_v2"}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    metrics, winner = [], None
    with threadpool_limits(limits=4):
        for split in (9, 10, 11, "scaffold"):
            names = ("tabicl-raw", "tabicl-log") if split == 9 else (winner,)
            if split == "scaffold":
                scaffold = scaffold_folds(scaffolds_of(data.train), 5)
                iterator = (fold_indices(scaffold, f) for f in range(5))
            else:
                iterator = KFold(5, shuffle=True, random_state=split).split(data.y)
            predictions = {name: np.full(len(data.y), np.nan) for name in names}
            folds = np.full(len(data.y), -1)
            for fold, (tr, va) in enumerate(iterator):
                a, b = features(data.train.iloc[tr], data.train.iloc[va], embeddings)
                for name in names:
                    predictions[name][va] = predict(a, data.y[tr], b, name.endswith("log"))
                folds[va] = fold
                elapsed = time.monotonic()-start
                print(f"split={split} fold={fold} elapsed={elapsed:.1f}s", flush=True)
                if elapsed > 600 or (split == 9 and fold == 0 and elapsed * 13 > 600):
                    manifest.update(status="stopped-budget", elapsed_seconds=elapsed)
                    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
                    pd.DataFrame({"row": np.arange(len(data.y)), "target": data.y, "fold": folds, **predictions}).to_csv(out/"partial-oof.csv", index=False)
                    return
            for name, p in predictions.items():
                error = (p-data.y)**2
                metrics.append({"split": split, "model": name, "mse": float(error.mean()),
                                "mse_without_max_diagnostic_only": float(np.delete(error, np.argmax(data.y)).mean()),
                                "max_label_prediction": float(p[np.argmax(data.y)]), "prediction_max": float(p.max())})
            pd.DataFrame({"row": np.arange(len(data.y)), "target": data.y, "fold": folds, **predictions}).to_csv(out/f"oof-{split}.csv", index=False)
            (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            print(metrics[-len(names):], flush=True)
            if split == 9:
                winner = min(metrics, key=lambda item: item["mse"])["model"]
        a, b = features(data.train, data.test, embeddings)
        p = predict(a, data.y, b, winner.endswith("log"))
        template = pd.read_csv("data/raw_v2/sample_submit.csv")
        result = submission(data.test["index"], p).set_index("index").loc[template["index"]].reset_index()
        result.to_csv(out/f"submission-{winner}.csv", index=False)
    manifest.update(status="completed", winner=winner, elapsed_seconds=time.monotonic()-start)
    checkpoint = Path(hf_hub_download("jingang/TabICL", manifest["checkpoint"], local_files_only=True))
    manifest.update(checkpoint_path=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
