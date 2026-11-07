"""Fixed two-part SI regressors; every label retained, no sentinel semantics."""
import json
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from vacv.dataset import load, submission
from vacv.pipeline import ContextEncoder, structure_matrix, scaffolds_of
from vacv.splits import scaffold_folds, fold_indices


def predict(train, y, query, cutoff):
    fields = ["SMILES", "Title", "strain", "Culture"]
    train, query = train[fields], query[fields]
    encoder = ContextEncoder().fit(train)
    x = np.hstack([encoder.transform(train), structure_matrix(train)])
    q = np.hstack([encoder.transform(query), structure_matrix(query)])
    # cutoff zero separates exact zeros; cutoff100 includes 100 in the upper bin.
    group = (y > 0) if cutoff == 0 else (y >= cutoff)
    predictions = []
    for seed in (0, 1):
        options = dict(n_estimators=400, min_samples_leaf=2, max_features=.5,
                       random_state=seed, n_jobs=2)
        if len(np.unique(group)) == 1:
            probability = np.full(len(query), float(group[0]))
        else:
            classifier = ExtraTreesClassifier(**options).fit(x, group)
            probability = classifier.predict_proba(q)[:, list(classifier.classes_).index(True)]
        means = []
        for value in (False, True):
            selected = group == value
            if selected.sum() < 4 or np.ptp(y[selected]) == 0:
                means.append(np.full(len(query), float(y[selected].mean()) if selected.any() else float(y.mean())))
            else:
                regressor = ExtraTreesRegressor(**options).fit(x[selected], y[selected])
                means.append(regressor.predict(q))
        predictions.append((1-probability)*means[0] + probability*means[1])
    p = np.maximum(0, np.mean(predictions, axis=0))
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite hurdle prediction")
    return p


def main():
    start = time.monotonic()
    out = Path("experiments/results/v2-hurdle")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    metrics, winner = [], None
    manifest = {"cutoffs": [0, 100], "discovery": 9, "confirmation": [10, 11],
                "train_sha256": hashlib.sha256(Path("data/raw_v2/train_v2.csv").read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "features": "fold-local ContextEncoder plus Morgan and MACCS", "target_version": "train_v2",
                "ensemble_seeds": [0, 1], "n_estimators": 400, "min_samples_leaf": 2,
                "classifier_class_weight": None, "small_group_fallback": "group mean if n<4; train mean if empty"}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with threadpool_limits(limits=2):
        for split in (9, 10, 11, "scaffold"):
            cutoffs = (0, 100) if split == 9 else (winner,)
            if split == "scaffold":
                scaffold = scaffold_folds(scaffolds_of(data.train), 5)
                iterator = (fold_indices(scaffold, f) for f in range(5))
            else:
                iterator = KFold(5, shuffle=True, random_state=split).split(data.y)
            predictions = {f"hurdle-{c}": np.full(len(data.y), np.nan) for c in cutoffs}
            folds = np.full(len(data.y), -1)
            for fold, (tr, va) in enumerate(iterator):
                for cutoff in cutoffs:
                    predictions[f"hurdle-{cutoff}"][va] = predict(data.train.iloc[tr], data.y[tr], data.train.iloc[va], cutoff)
                folds[va] = fold
                elapsed = time.monotonic()-start
                print(f"split={split} fold={fold} elapsed={elapsed:.1f}s", flush=True)
                if elapsed > 600 or (split == 9 and fold == 0 and elapsed*13 > 600):
                    manifest.update(status="stopped-budget", elapsed_seconds=elapsed)
                    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
                    pd.DataFrame({"row": np.arange(len(data.y)), "target": data.y, "fold": folds, **predictions}).to_csv(out/"partial-oof.csv", index=False)
                    return
            for name, p in predictions.items():
                error = (p-data.y)**2
                metrics.append({"split": split, "model": name, "mse": float(error.mean()),
                                "mse_without_max_diagnostic_only": float(np.delete(error, np.argmax(data.y)).mean()),
                                "tail_mse": float(error[data.y >= 1000].mean()),
                                "non_tail_mse": float(error[data.y < 1000].mean()),
                                "max_label_prediction": float(p[np.argmax(data.y)]), "prediction_max": float(p.max())})
            pd.DataFrame({"row": np.arange(len(data.y)), "target": data.y, "fold": folds, **predictions}).to_csv(out/f"oof-{split}.csv", index=False)
            (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            print(metrics[-len(cutoffs):], flush=True)
            if split == 9:
                winner = int(min(metrics, key=lambda item: item["mse"])["model"].split("-")[1])
        p = predict(data.train, data.y, data.test, winner)
        template = pd.read_csv("data/raw_v2/sample_submit.csv")
        result = submission(data.test["index"], p).set_index("index").loc[template["index"]].reset_index()
        result.to_csv(out/f"submission-hurdle-{winner}.csv", index=False)
    manifest.update(status="completed", winner=winner, elapsed_seconds=time.monotonic()-start)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
