"""Paired measured-only external-data ablation; all competition molecules excluded."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from vacv.curate import standardize
from vacv.dataset import load, submission
from vacv.pipeline import structure_matrix


def main():
    out = Path("experiments/results/v2-external")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    external = pd.read_csv("data/processed/vacv_si_measured.csv",
                           usecols=["smiles", "si", "source_db", "si_quality"])
    def key(s):
        k = standardize(s).inchikey
        return k[:14] if k else None
    blocked = {key(s) for s in pd.concat([data.train.SMILES, data.test.SMILES]).unique()}
    external["key"] = [key(s) for s in external.smiles]
    external = external[external.source_db.isin(["ChEMBL", "ChEMBL|PubChem BioAssay"]) & external.si_quality.eq("measured")
                        & external.key.notna() & ~external.key.isin(blocked)
                        & np.isfinite(external.si) & external.si.ge(0)]
    external = external.groupby("key", as_index=False).agg(smiles=("smiles", "first"), si=("si", "median"))
    if not len(external):
        raise ValueError("No disjoint measured ChEMBL rows")
    x = structure_matrix(data.train); xt = structure_matrix(data.test)
    xe = structure_matrix(external.rename(columns={"smiles": "SMILES"}))
    ye = np.log1p(external.si.to_numpy())
    y = data.y
    def predict(tr, q, augment):
        xx, yy = x[tr], np.log1p(y[tr])
        weights = np.ones(len(tr))
        if augment:
            xx = np.vstack([xx, xe]); yy = np.r_[yy, ye]
            weights = np.r_[weights, np.full(len(ye), .25)]
        predictions = []
        for seed in range(3):
            model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                        max_features=.5, random_state=seed, n_jobs=2)
            model.fit(xx, yy, sample_weight=weights)
            predictions.append(model.predict(q))
        return np.maximum(0, np.expm1(np.mean(predictions, axis=0)))
    metrics = []
    with threadpool_limits(limits=2):
        for split in (6, 7, 8):
            pred = {n: np.full(len(y), np.nan) for n in ("structure-log", "structure-log-external")}
            folds = np.full(len(y), -1)
            for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=split).split(y)):
                for i, name in enumerate(pred):
                    pred[name][va] = predict(tr, x[va], bool(i))
                folds[va] = fold
            for name, p in pred.items():
                metrics.append({"split": split, "model": name, "mse": float(np.mean((p-y)**2))})
            pd.DataFrame({"index": np.arange(len(y)), "fold": folds, **pred}).to_csv(out/f"oof-{split}.csv", index=False)
            (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            print(metrics[-2:], flush=True)
        template = pd.read_csv("data/raw_v2/sample_submit.csv")
        for i, name in enumerate(("structure-log", "structure-log-external")):
            result = submission(data.test["index"], predict(np.arange(len(y)), xt, bool(i)))
            result = result.set_index("index").loc[template["index"]].reset_index()
            result.to_csv(out/f"submission-{name}.csv", index=False)
    manifest = {"external_rows": len(external), "excluded_competition_keys": len(blocked),
                "overlap_after_filter": len(set(external.key) & blocked),
                "source": "data/processed/vacv_si_measured.csv", "external_weight": .25,
                "target_version": "train_v2", "source_sha256": hashlib.sha256(
                    Path("data/processed/vacv_si_measured.csv").read_bytes()).hexdigest()}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
