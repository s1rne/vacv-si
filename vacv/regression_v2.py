"""Regression of official v2 SI, including zeros and the full positive tail."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from .pipeline import ContextEncoder, structure_matrix
from .kernel import KernelSelectivityModel

FIELDS = ["index", "SMILES", "Title", "strain", "Culture"]
PROFILES = ("constant", "et-context-raw", "et-context-log", "et-blend-raw",
            "et-blend-log", "kernel-raw", "kernel-log", "catboost-raw")


def allowed(frame):
    return frame[[c for c in FIELDS if c in frame]].copy()


class RegressionV2:
    def __init__(self, profile="et-blend-log"):
        if profile not in PROFILES:
            raise ValueError("Unknown regression profile")
        self.profile = profile

    def fit(self, frame, y):
        frame = allowed(frame)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not len(y) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Aligned finite nonnegative SI targets required")
        self.log_ = self.profile.endswith("-log")
        target = np.log1p(y) if self.log_ else y
        self.prior_ = float(target.mean())
        if self.profile == "constant":
            return self
        with threadpool_limits(limits=2):
            if self.profile.startswith("kernel"):
                self.inputs_ = KernelSelectivityModel._inputs(frame)
                matrix = KernelSelectivityModel._kernel(self.inputs_, self.inputs_)
                self.coefficients_ = np.linalg.solve(matrix + np.eye(len(frame)), target-self.prior_)
            else:
                self.encoder_ = ContextEncoder().fit(frame)
                context = self.encoder_.transform(frame)
                structure = structure_matrix(frame)
                if self.profile == "catboost-raw":
                    from catboost import CatBoostRegressor
                    self.cat_ = CatBoostRegressor(iterations=500, depth=4, learning_rate=.03,
                        l2_leaf_reg=5, loss_function="RMSE", verbose=False, thread_count=2,
                        random_seed=0, allow_writing_files=False).fit(np.column_stack([context, structure]), target)
                else:
                    self.context_, self.structure_ = [], []
                    for seed in range(3):
                        opts = dict(n_estimators=800, min_samples_leaf=2, max_features=.5, n_jobs=1, random_state=seed)
                        self.context_.append(ExtraTreesRegressor(**opts).fit(context, target))
                        if self.profile.startswith("et-blend"):
                            self.structure_.append(ExtraTreesRegressor(**opts).fit(structure, target))
        return self

    def predict(self, frame):
        frame = allowed(frame)
        with threadpool_limits(limits=2):
            if self.profile == "constant":
                value = np.full(len(frame), self.prior_)
            elif self.profile.startswith("kernel"):
                matrix = KernelSelectivityModel._kernel(KernelSelectivityModel._inputs(frame), self.inputs_)
                value = self.prior_ + matrix @ self.coefficients_
            elif self.profile == "catboost-raw":
                value = self.cat_.predict(np.column_stack([self.encoder_.transform(frame), structure_matrix(frame)]))
            else:
                context = self.encoder_.transform(frame)
                value = np.mean([m.predict(context) for m in self.context_], axis=0)
                if self.structure_:
                    structure = structure_matrix(frame)
                    value = .5*(value + np.mean([m.predict(structure) for m in self.structure_], axis=0))
        value = np.maximum(0, np.expm1(np.minimum(value, 20)) if self.log_ else value)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite regression prediction")
        return value
