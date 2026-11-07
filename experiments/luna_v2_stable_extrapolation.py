"""Fold-local stabilized feature extrapolation for official v2 SI labels."""
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
from vacv.dataset import FORBIDDEN, load, submission
from vacv.features import DESCRIPTORS, parse
from vacv.pipeline import ContextEncoder, scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = ("ridge-raw", "ridge-sqrt", "kernel-raw", "kernel-sqrt")
FIELDS = ["SMILES", "Title", "strain", "Culture"]


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


def stable_descriptor_features(train, query):
    """Signed-log transform and train-only 1/99% clipping for descriptors."""
    train = np.sign(train) * np.log1p(np.abs(train))
    query = np.sign(query) * np.log1p(np.abs(query))
    low = np.nanquantile(train, 0.01, axis=0)
    high = np.nanquantile(train, 0.99, axis=0)
    low = np.where(np.isfinite(low), low, 0.0)
    high = np.where(np.isfinite(high), high, 0.0)
    return np.clip(train, low, high), np.clip(query, low, high)


def features(train, query, dtrain, dquery, embeddings):
    train, query = train[FIELDS], query[FIELDS]
    da, db = stable_descriptor_features(dtrain, dquery)
    encoder = ContextEncoder().fit(train)
    pca = PCA(n_components=min(16, len(train) - 1), random_state=0)
    a = pca.fit_transform(np.stack([embeddings[s] for s in train.SMILES]))
    b = pca.transform(np.stack([embeddings[s] for s in query.SMILES]))
    x = np.hstack([da, encoder.transform(train), a])
    q = np.hstack([db, encoder.transform(query), b])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
    x, q = imputer.transform(x), imputer.transform(q)
    scaler = StandardScaler().fit(x)
    return scaler.transform(x), scaler.transform(q)


def predict(x, y, q, name):
    target = np.sqrt(y) if name.endswith("sqrt") else y
    if name.startswith("ridge"):
        value = Ridge(alpha=10.0).fit(x, target).predict(q)
    else:
        prior = float(target.mean())
        model = KernelRidge(alpha=1.0, kernel="polynomial", degree=2,
                            gamma=1.0 / x.shape[1], coef0=1.0)
        value = prior + model.fit(x, target - prior).predict(q)
    value = np.maximum(0.0, value)
    if name.endswith("sqrt"):
        value = value**2
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite predictions from {name}")
    return value


def evaluate(data, dx, embeddings, split, names):
    if split == "scaffold":
        groups = scaffold_folds(scaffolds_of(data.train), 5)
        iterator = (fold_indices(groups, f) for f in range(5))
    else:
        iterator = KFold(5, shuffle=True, random_state=split).split(data.y)
    pred = {name: np.full(len(data.y), np.nan) for name in names}
    folds = np.full(len(data.y), -1, dtype=int)
    for fold, (tr, va) in enumerate(iterator):
        x, q = features(data.train.iloc[tr], data.train.iloc[va], dx[tr], dx[va], embeddings)
        for name in names:
            pred[name][va] = predict(x, data.y[tr], q, name)
        folds[va] = fold
    rows = []
    for name, p in pred.items():
        err = (p - data.y) ** 2
        rows.append({"split": split, "model": name, "mse": float(err.mean()),
                     "non_tail_mse": float(err[data.y < 1000].mean()),
                     "tail_mse": float(err[data.y >= 1000].mean()),
                     "max_label_prediction": float(p[np.argmax(data.y)]),
                     "prediction_max": float(p.max())})
    frame = pd.DataFrame({"row": np.arange(len(data.y)), "target": data.y,
                          "fold": folds, **pred})
    if not np.isfinite(frame[list(names)]).all().all() or (frame[list(names)] < 0).any().any():
        raise ValueError("OOF predictions must be finite and nonnegative")
    return rows, frame


def main():
    start = time.monotonic()
    out = Path("experiments/results/luna-v2-stable-extrapolation")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    if set(FORBIDDEN).intersection(data.train.columns):
        raise ValueError("Forbidden competition columns present in loaded training frame")
    embeddings, embed_manifest = load_embeddings("data/raw/models/chemeleon_features.npz")
    dx = physicochem(data.train)
    metrics = []
    winner = None
    with threadpool_limits(limits=2):
        for split in (18, 19, 20, "scaffold"):
            names = PROFILES if split == 18 else (winner,)
            rows, frame = evaluate(data, dx, embeddings, split, names)
            metrics.extend(rows)
            frame.to_csv(out / f"oof-{split}.csv", index=False)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(rows, flush=True)
            if split == 18:
                winner = min(rows, key=lambda row: row["mse"])["model"]
        x, q = features(data.train, data.test, dx, physicochem(data.test), embeddings)
        p = predict(x, data.y, q, winner)
        template = pd.read_csv("data/raw_v2/sample_submit.csv")
        result = submission(data.test["index"], p).set_index("index").loc[template["index"]].reset_index()
        result.to_csv(out / f"candidate-{winner}.csv", index=False)
    confirmation = [row["mse"] for row in metrics if row["split"] in (19, 20)]
    manifest = {"profiles": PROFILES, "discovery_seed": 18, "confirmation_seeds": [19, 20],
                "winner": winner,
                "passes_3m_both_confirmations": bool(len(confirmation) == 2 and max(confirmation) <= 3e6),
                "scaffold_mse": next(row["mse"] for row in metrics if row["split"] == "scaffold"),
                "candidate_submission_status": "gate_pass" if len(confirmation) == 2 and max(confirmation) <= 3e6 else "GATE_FAIL_DO_NOT_SUBMIT",
                "target_version": "train_v2", "all_320_targets_used": True,
                "no_target_clipping": True, "feature_transform": "signed_log1p_descriptors_then_training_1_99_quantile_clip",
                "train_sha256": hashlib.sha256(Path("data/raw_v2/train_v2.csv").read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "embedding_sha256": embed_manifest["embedding_sha256"],
                "elapsed_seconds": time.monotonic() - start}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
