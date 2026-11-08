"""Inner-crossfit log-location/variance correction; approximate lognormal mean."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from vacv.dataset import load, submission
from vacv.pipeline import ContextEncoder, structure_matrix, scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

FIELDS = ["SMILES", "Title", "strain", "Culture"]
PROFILES = {"log-baseline": None, "variance25": .25, "variance75": .75}


def matrix(frame, encoder):
    frame = frame[FIELDS]
    return np.column_stack([encoder.transform(frame), structure_matrix(frame)])


def fit_mean(frame, latent, trees=400):
    frame = frame[FIELDS]
    encoder = ContextEncoder().fit(frame)
    model = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=2,
        max_features=.5, n_jobs=1, random_state=0).fit(matrix(frame, encoder), latent)
    return encoder, model


def mean_predict(fitted, frame):
    encoder, model = fitted
    return model.predict(matrix(frame, encoder))


def crossfit_residuals(frame, latent, trees=400):
    """Each residual is predicted without its own label; only outer train enters."""
    predicted = np.full(len(latent), np.nan)
    for train, valid in KFold(3, shuffle=True, random_state=42).split(latent):
        fitted = fit_mean(frame.iloc[train], latent[train], trees)
        predicted[valid] = mean_predict(fitted, frame.iloc[valid])
    if not np.isfinite(predicted).all():
        raise ValueError("Incomplete inner OOF")
    return (latent-predicted)**2


class DistributionalRegression:
    def __init__(self, trees=400):
        self.trees = trees

    def fit(self, frame, y):
        frame = frame[FIELDS]
        y = np.asarray(y, dtype=float)
        if len(y) < 3 or y.shape != (len(frame),) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("At least three aligned nonnegative targets required")
        latent = np.log1p(y)
        residual_squared = crossfit_residuals(frame, latent, self.trees)
        self.mean_ = fit_mean(frame, latent, self.trees)
        self.global_variance_ = float(residual_squared.mean())
        self.variance_ = ExtraTreesRegressor(n_estimators=self.trees,
            min_samples_leaf=5, max_features=.5, random_state=0, n_jobs=1)
        self.variance_.fit(matrix(frame, self.mean_[0]), residual_squared)
        return self

    def predict_all(self, frame):
        frame = frame[FIELDS]
        mean = mean_predict(self.mean_, frame)
        conditional = np.maximum(0, self.variance_.predict(matrix(frame, self.mean_[0])))
        outputs = {}
        for name, shrink in PROFILES.items():
            variance = 0 if shrink is None else shrink*conditional+(1-shrink)*self.global_variance_
            with np.errstate(over="raise", invalid="raise"):
                value = np.maximum(0, np.expm1(mean+.5*variance))
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite unbounded distributional prediction")
            outputs[name] = value
        return outputs


def main():
    start = time.monotonic()
    out = Path("experiments/results/v3-distributional")
    out.mkdir(parents=True, exist_ok=False)
    data = load(Path("data/raw_v2"), require_v2=True)
    manifest = {"profiles": PROFILES, "discovery": 24, "confirmation": [25, 26],
                "inner_folds": 3, "inner_seed": 42, "trees": 400,
                "assumption": "Approximate lognormal 1+SI, including zeros; not proven calibrated uncertainty",
                "target_rows": len(data.y), "target_max": float(data.y.max()),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "train_sha256": hashlib.sha256(Path("data/raw_v2/train_v2.csv").read_bytes()).hexdigest()}
    metrics = []
    with threadpool_limits(limits=2):
        for split in (24, 25, 26, "scaffold"):
            if split == "scaffold":
                sf = scaffold_folds(scaffolds_of(data.train), 5)
                iterator = [fold_indices(sf, f) for f in range(5)]
            else:
                iterator = KFold(5, shuffle=True, random_state=split).split(data.y)
            names = list(PROFILES) if split == 24 else list(dict.fromkeys(["log-baseline", winner]))
            predictions = {name: np.full(len(data.y), np.nan) for name in names}
            folds = np.full(len(data.y), -1)
            for fold, (train, valid) in enumerate(iterator):
                fitted = DistributionalRegression().fit(data.train.iloc[train], data.y[train])
                p = fitted.predict_all(data.train.iloc[valid])
                for name in names:
                    predictions[name][valid] = p[name]
                folds[valid] = fold
                print(f"split={split} fold={fold} elapsed={time.monotonic()-start:.1f}s", flush=True)
            for name, p in predictions.items():
                error = (p-data.y)**2
                if not np.isfinite(error).all():
                    raise ValueError("Incomplete or nonfinite full OOF errors")
                metrics.append({"split": split, "model": name, "mse": float(error.mean()),
                                "maximum_target_prediction": float(p[data.y.argmax()]),
                                "prediction_max": float(p.max())})
            pd.DataFrame({"row": np.arange(len(data.y)), "target": data.y, "fold": folds,
                          **predictions}).to_csv(out/f"oof-{split}.csv", index=False)
            (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            if split == 24:
                winner = min(metrics, key=lambda item: item["mse"])["model"]
                manifest["frozen_winner"] = winner
            (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        confirmation = [r["mse"] for r in metrics if r["split"] in (25, 26) and r["model"] == winner]
        manifest["gate_pass"] = len(confirmation) == 2 and max(confirmation) <= 3_000_000
        if manifest["gate_pass"]:
            fitted = DistributionalRegression().fit(data.train, data.y)
            result = submission(data.test["index"], fitted.predict_all(data.test)[winner])
            template = pd.read_csv("data/raw_v2/sample_submit.csv")
            result = result.set_index("index").loc[template["index"]].reset_index()
            result.to_csv(out/"candidate.csv", index=False)
    manifest.update(status="completed", elapsed_seconds=time.monotonic()-start)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
