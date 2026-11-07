"""Fold-local bounded power regression of all official SI targets.

Power zero denotes log1p. Predictions stay inside the training target range;
there is no label clipping, smearing estimated on validation, or lookup override.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from .regression_v2 import allowed
from .pipeline import ContextEncoder, structure_matrix

CONFIGS = {
    'log': ((0., 1.),),
    'quarter': ((.25, 1.),),
    'sqrt': ((.5, 1.),),
    'raw': ((1., 1.),),
    'log90-raw10': ((0., .9), (1., .1)),
    'sqrt50-log50': ((.5, .5), (0., .5)),
}


def transform(y, power):
    return np.log1p(y) if power == 0 else np.power(1 + y, power) - 1


def inverse(z, power):
    return np.expm1(z) if power == 0 else np.power(1 + z, 1 / power) - 1


class PowerRegression:
    def __init__(self, config='sqrt', n_estimators=800, seeds=(0, 1, 2)):
        if config not in CONFIGS:
            raise ValueError('Unknown power configuration')
        self.config, self.n_estimators, self.seeds = config, n_estimators, seeds

    def fit(self, frame, y):
        frame = allowed(frame)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not len(y) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('Aligned finite nonnegative targets required')
        self.maximum_ = float(y.max())
        self.encoder_ = ContextEncoder().fit(frame)
        features = (self.encoder_.transform(frame), structure_matrix(frame))
        self.models_ = {}
        with threadpool_limits(limits=2):
            for power, _ in CONFIGS[self.config]:
                target = transform(y, power)
                self.models_[power] = [[ExtraTreesRegressor(n_estimators=self.n_estimators,
                    min_samples_leaf=2, max_features=.5, random_state=seed, n_jobs=1
                    ).fit(x, target) for seed in self.seeds] for x in features]
        return self

    def predict(self, frame):
        frame = allowed(frame)
        features = (self.encoder_.transform(frame), structure_matrix(frame))
        prediction = np.zeros(len(frame))
        for power, weight in CONFIGS[self.config]:
            transformed = np.mean([m.predict(x) for x, models in zip(features, self.models_[power])
                                   for m in models], axis=0)
            transformed = np.clip(transformed, 0, transform(self.maximum_, power))
            prediction += weight * inverse(transformed, power)
        return np.clip(prediction, 0, self.maximum_)
