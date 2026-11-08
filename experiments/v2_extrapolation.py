"""Four fixed extrapolating regressors; no target-range upper clipping."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from vacv.checkpoint import load_embeddings
from vacv.dataset import load, submission
from vacv.features import DESCRIPTORS, parse
from vacv.pipeline import ContextEncoder, scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = ("ridge-raw", "ridge-sqrt", "poly2-raw", "poly2-sqrt")


def physicochem(frame):
    rows = []
    for smiles in frame.SMILES:
        mol = parse(smiles)
        if mol is None:
            rows.append(np.full(len(DESCRIPTORS), np.nan))
        else:
            values = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)
            rows.append([values[name] for name in DESCRIPTORS])
    x = np.asarray(rows, dtype=float)
    x[~np.isfinite(x)] = np.nan
    return x


def features(train, query, dtrain, dquery, embeddings):
    fields = ["SMILES", "Title", "strain", "Culture"]
    train, query = train[fields], query[fields]
    encoder = ContextEncoder().fit(train)
    pca = PCA(n_components=min(16, len(train)-1), random_state=0)
    a = pca.fit_transform(np.stack([embeddings[s] for s in train.SMILES]))
    b = pca.transform(np.stack([embeddings[s] for s in query.SMILES]))
    x = np.hstack([dtrain, encoder.transform(train), a])
    q = np.hstack([dquery, encoder.transform(query), b])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
    x, q = imputer.transform(x), imputer.transform(q)
    scaler = StandardScaler().fit(x)
    return scaler.transform(x), scaler.transform(q)


def predict(x, y, q, name):
    target = np.sqrt(y) if name.endswith("sqrt") else y
    if name.startswith("ridge"):
        value = Ridge(alpha=10).fit(x, target).predict(q)
    else:
        prior = float(target.mean())
        model = KernelRidge(alpha=1., kernel="polynomial", degree=2,
                            gamma=1./x.shape[1], coef0=1.)
        value = prior + model.fit(x, target-prior).predict(q)
    value = np.maximum(0, value)
    value = value**2 if name.endswith("sqrt") else value
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite extrapolating prediction")
    return value


def main():
    start = time.monotonic()
    out = Path("experiments/results/v2-extrapolation")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    embeddings, embed_manifest = load_embeddings("data/raw/models/chemeleon_features.npz")
    dx, dq = physicochem(data.train), physicochem(data.test)
    metrics, winner = [], None
    with threadpool_limits(limits=2):
        for split in (15, 16, 17, "scaffold"):
            names = PROFILES if split == 15 else (winner,)
            if split == "scaffold":
                scaffold = scaffold_folds(scaffolds_of(data.train), 5)
                iterator = (fold_indices(scaffold, f) for f in range(5))
            else:
                iterator = KFold(5, shuffle=True, random_state=split).split(data.y)
            pred = {name: np.full(len(data.y), np.nan) for name in names}
            folds = np.full(len(data.y), -1)
            for fold, (tr, va) in enumerate(iterator):
                x, q = features(data.train.iloc[tr], data.train.iloc[va], dx[tr], dx[va], embeddings)
                for name in names:
                    pred[name][va] = predict(x, data.y[tr], q, name)
                folds[va] = fold
            for name, p in pred.items():
                err = (p-data.y)**2
                maxrow = np.argmax(data.y)
                metrics.append({"split": split, "model": name, "mse": float(err.mean()),
                                "mse_without_max_diagnostic_only": float(np.delete(err, maxrow).mean()),
                                "max_label_prediction": float(p[maxrow]), "prediction_max": float(p.max()),
                                "non_tail_mse": float(err[data.y < 1000].mean()),
                                "tail_mse": float(err[data.y >= 1000].mean())})
            pd.DataFrame({"row": np.arange(len(data.y)), "target": data.y, "fold": folds, **pred}).to_csv(out/f"oof-{split}.csv", index=False)
            (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            print(metrics[-len(names):], flush=True)
            if split == 15:
                winner = min(metrics, key=lambda item: item["mse"])["model"]
        x, q = features(data.train, data.test, dx, dq, embeddings)
        p = predict(x, data.y, q, winner)
        template = pd.read_csv("data/raw_v2/sample_submit.csv")
        result = submission(data.test["index"], p).set_index("index").loc[template["index"]].reset_index()
        result.to_csv(out/f"submission-{winner}.csv", index=False)
    confirmation = [m["mse"] for m in metrics if m["split"] in (16, 17)]
    manifest = {"profiles": PROFILES, "discovery": 15, "confirmation": [16, 17], "winner": winner,
                "passes_3m_both_confirmation": bool(len(confirmation) == 2 and max(confirmation) <= 3e6),
                "target_version": "train_v2", "no_upper_target_clipping": True,
                "train_sha256": hashlib.sha256(Path("data/raw_v2/train_v2.csv").read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "embedding_sha256": embed_manifest["embedding_sha256"], "elapsed_seconds": time.monotonic()-start}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
