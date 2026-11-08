"""Frozen six-profile experiment: PSD structure x assay-context product kernel."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v2_analog import diagnostics, fingerprint, tanimoto
from experiments.v2_context_analog import contexts, same_context, predict as analog
from vacv.dataset import load, submission
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = {f"product-{scale}-{alpha:g}": (scale, alpha)
            for scale in ("raw", "sqrt") for alpha in (.01, .1, 1.)}
CONTROL = "context-raw70"


def product_kernel(similarity, context_match):
    return similarity ** 2 * (.1 + .9 * context_match)


def predict(train_kernel, query_kernel, train_y, profile):
    """Fit using train labels only; consistently center train and query kernels."""
    scale, alpha = PROFILES[profile]
    target = train_y / 1000 if scale == "raw" else np.sqrt(train_y)
    col_mean = train_kernel.mean(axis=0)
    grand_mean = float(train_kernel.mean())
    centered = train_kernel - col_mean[None, :] - col_mean[:, None] + grand_mean
    query_centered = query_kernel - query_kernel.mean(axis=1, keepdims=True) - col_mean[None, :] + grand_mean
    mean = float(target.mean())
    coefficients = np.linalg.solve(centered + alpha * np.eye(len(target)), target - mean)
    transformed = np.maximum(0, mean + query_centered @ coefficients)
    result = transformed * 1000 if scale == "raw" else transformed ** 2
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite product-kernel predictions")
    return result


def fold_predictions(kernel, y, train, valid, profile):
    return predict(kernel[np.ix_(train, train)], kernel[np.ix_(valid, train)], y[train], profile)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path("experiments/results/v4-context-kernel")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    y = data.y
    x = fingerprint(data.train)
    sim = tanimoto(x, x)
    ctx = contexts(data.train)
    match = same_context(ctx, ctx)
    kernel = product_kernel(sim, match)
    metrics, winner = [], None
    with threadpool_limits(limits=2):
        for split in (30, 31, 32, "scaffold"):
            names = tuple(PROFILES) if split == 30 else (winner,)
            if split == "scaffold":
                groups = scaffold_folds(scaffolds_of(data.train), 5)
                iterator = (fold_indices(groups, f) for f in range(5))
            else:
                iterator = KFold(5, shuffle=True, random_state=split).split(y)
            predictions = {name: np.full(len(y), np.nan) for name in (*names, CONTROL)}
            folds = np.full(len(y), -1)
            seen = np.zeros(len(y), dtype=int)
            for fold, (train, valid) in enumerate(iterator):
                for name in names:
                    predictions[name][valid] = fold_predictions(kernel, y, train, valid, name)
                predictions[CONTROL][valid] = analog(sim[np.ix_(valid, train)], match[np.ix_(valid, train)], y[train], CONTROL)[0]
                folds[valid] = fold
                seen[valid] += 1
            assert (seen == 1).all() and len(y) == 320
            for name, pred in predictions.items():
                assert np.isfinite(pred).all()
                metrics.append({"split": split, "model": name, **diagnostics(y, pred)})
            if split == 30:
                winner = min((m for m in metrics if m["model"] in PROFILES), key=lambda m: m["mse"])["model"]
                (out / "frozen-selection.json").write_text(json.dumps({"discovery_seed": 30, "winner": winner, "profiles": PROFILES}, indent=2) + "\n")
            pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds, **predictions}).to_csv(out / f"oof-{split}.csv", index=False)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics[-len(predictions):]), flush=True)
        confirmation = [next(m["mse"] for m in metrics if m["split"] == split and m["model"] == winner) for split in (31, 32)]
        controls = [next(m["mse"] for m in metrics if m["split"] == split and m["model"] == CONTROL) for split in (31, 32)]
        hard_gate = all(m <= 3_000_000 for m in confirmation)
        local_gate = np.mean(confirmation) < 4_000_000 and all(a < b for a, b in zip(confirmation, controls))
        if hard_gate:
            xt = fingerprint(data.test)
            test_kernel = product_kernel(tanimoto(xt, x), same_context(contexts(data.test), ctx))
            p = predict(kernel, test_kernel, y, winner)
            template = pd.read_csv("data/raw_v2/sample_submit.csv")
            result = submission(data.test["index"], p).set_index("index").loc[template["index"]].reset_index()
            result.to_csv(out / f"submission-{winner}.csv", index=False)
    sources = [__file__, "experiments/v2_analog.py", "experiments/v2_context_analog.py", "vacv/dataset.py", "vacv/pipeline.py", "vacv/splits.py"]
    manifest = {"profiles": PROFILES, "discovery": 30, "confirmation": [31, 32], "winner": winner,
                "selection": "minimum full raw MSE on discovery only", "external_data": False,
                "target_version": "train_v2", "rows": len(y), "target_max": float(y.max()),
                "target_cap": None, "prediction_cap": None, "kernel": "Tanimoto^2 * (0.1 + 0.9 * same known strain,culture)",
                "intercept": "unpenalized train mean with consistently centered kernel",
                "confirmation_mse": confirmation, "paired_control_mse": controls,
                "confirmation_mean": float(np.mean(confirmation)), "local_4m_gate": bool(local_gate),
                "hard_upload_3m_each_gate": hard_gate, "submitted": False,
                "train_sha256": sha("data/raw_v2/train_v2.csv"),
                "source_hashes": {str(p): sha(p) for p in sources},
                "oof_hashes": {p.name: sha(p) for p in sorted(out.glob("oof-*.csv"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
