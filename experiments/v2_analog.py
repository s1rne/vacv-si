"""Four predeclared molecular analog regressors, official-v2 full-tail CV."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from vacv.curate import standardize
from vacv.dataset import load, submission
from vacv.pipeline import structure_matrix, scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

CONFIGS = ("knn8-raw", "knn8-log", "knn8-log-external", "local-krr24-raw")


def fingerprint(frame):
    return (structure_matrix(frame)[:, :2048] > 0).astype(float)


def tanimoto(a, b):
    dot = a @ b.T
    denominator = a.sum(1)[:, None] + b.sum(1)[None, :] - dot
    return np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)


def predict(name, similarity, train_similarity, y, external_count=0):
    """Query-local hyperparameters are fixed and never fitted to validation labels."""
    log = "log" in name
    target = np.log1p(y) if log else y
    result = []
    for s in similarity:
        k = min(24 if name.startswith("local") else 8, len(y))
        ix = np.argsort(-s, kind="stable")[:k]
        if name.startswith("local"):
            prior = float(np.median(target[ix]))
            coefficients = np.linalg.solve(train_similarity[np.ix_(ix, ix)] ** 2
                                           + .5 * np.eye(k), target[ix] - prior)
            value = prior + (s[ix] ** 2) @ coefficients
        else:
            weights = s[ix] ** 8
            if external_count:
                weights *= np.where(ix >= len(y) - external_count, .25, 1.)
            value = weights @ target[ix] / weights.sum() if weights.sum() > 0 else np.median(target)
        result.append(value)
    result = np.asarray(result)
    return np.maximum(0., np.expm1(result) if log else result)


def diagnostics(y, p):
    tail = y >= 1000
    maxrow = int(np.argmax(y))
    return {"mse": float(np.mean((p-y)**2)), "mse_without_max_diagnostic_only": float(np.mean(np.delete((p-y)**2, maxrow))),
            "tail_mse": float(np.mean((p[tail]-y[tail])**2)), "non_tail_mse": float(np.mean((p[~tail]-y[~tail])**2)),
            "max_label_prediction": float(p[maxrow]), "prediction_max": float(p.max()), "prediction_mean": float(p.mean())}


def main():
    out = Path("experiments/results/v2-analog")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    source = Path("data/processed/vacv_si_measured.csv")
    ext = pd.read_csv(source, usecols=["smiles", "si", "source_db", "si_quality"])
    def key(smiles):
        value = standardize(smiles).inchikey
        return value[:14] if value else None
    blocked = {key(s) for s in pd.concat([data.train.SMILES, data.test.SMILES]).unique()}
    ext["key"] = [key(s) for s in ext.smiles]
    ext = ext[ext.source_db.isin(["ChEMBL", "ChEMBL|PubChem BioAssay"]) & ext.si_quality.eq("measured")
              & ext.key.notna() & ~ext.key.isin(blocked) & np.isfinite(ext.si) & ext.si.ge(0)]
    ext = ext.groupby("key", as_index=False).agg(smiles=("smiles", "first"), si=("si", "median"))
    x, xt, xe = fingerprint(data.train), fingerprint(data.test), fingerprint(ext.rename(columns={"smiles": "SMILES"}))
    allx = np.vstack([x, xe])
    similarity = tanimoto(allx, allx)
    test_similarity = tanimoto(xt, allx)
    y, ye = data.y, ext.si.to_numpy()
    metrics = []
    with threadpool_limits(limits=2):
        for split in (9, 10, 11, "scaffold"):
            folds = np.full(len(y), -1)
            if split == "scaffold":
                scaffold = scaffold_folds(scaffolds_of(data.train), 5)
                iterator = (fold_indices(scaffold, f) for f in range(5))
            else:
                iterator = KFold(5, shuffle=True, random_state=split).split(y)
            predictions = {name: np.full(len(y), np.nan) for name in CONFIGS}
            for fold, (tr, va) in enumerate(iterator):
                folds[va] = fold
                for name in CONFIGS:
                    augmented = name.endswith("external")
                    ix = np.r_[tr, np.arange(len(y), len(allx))] if augmented else tr
                    labels = np.r_[y[tr], ye] if augmented else y[tr]
                    predictions[name][va] = predict(name, similarity[np.ix_(va, ix)], similarity[np.ix_(ix, ix)], labels,
                                                    len(ye) if augmented else 0)
            for name, p in predictions.items():
                if not np.isfinite(p).all():
                    raise ValueError("Nonfinite OOF")
                metrics.append({"split": split, "model": name, **diagnostics(y, p)})
            pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds, **predictions}).to_csv(out/f"oof-{split}.csv", index=False)
            (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            print(metrics[-4:], flush=True)
        template = pd.read_csv("data/raw_v2/sample_submit.csv")
        for name in CONFIGS:
            augmented = name.endswith("external")
            ix = np.arange(len(allx) if augmented else len(y))
            p = predict(name, test_similarity[:, ix], similarity[np.ix_(ix, ix)], np.r_[y, ye] if augmented else y,
                        len(ye) if augmented else 0)
            result = submission(data.test["index"], p).set_index("index").loc[template["index"]].reset_index()
            result.to_csv(out/f"submission-{name}.csv", index=False)
    manifest = {"configs": CONFIGS, "discovery": 9, "confirmation": [10, 11], "target_version": "train_v2",
                "external_rows": len(ext), "external_overlap": len(set(ext.key) & blocked),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "selection_metric": "full raw MSE, all targets retained", "features": "Morgan binary radius2 2048; structure only"}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
