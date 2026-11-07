"""Fixed, clean-input hybrid with train-validated sentinel sharpening."""
import numpy as np
from threadpoolctl import threadpool_limits

from .clean_features import attach_clean
from .inference import PROFILES
from .kernel import KernelSelectivityModel
from .pipeline import INACTIVE, SENTINEL, SelectivityModel


def sharpen_probability(probability, exponent=1.2):
    probability = np.asarray(probability, dtype=float)
    if (not np.isfinite(probability).all() or (probability < 0).any()
            or (probability > 1).any() or not np.isfinite(exponent) or exponent <= 0):
        raise ValueError("Finite probabilities in [0,1] and positive exponent required")
    left, right = probability ** exponent, (1 - probability) ** exponent
    return left / (left + right)


class RecoveryModel:
    """Only allowed context/structure and unsupervised external presence features."""

    def __init__(self, seeds=7, exponent=1.2):
        if not isinstance(seeds, int) or seeds < 1:
            raise ValueError("seeds must be a positive integer")
        sharpen_probability([0.5], exponent)
        self.seeds, self.exponent = seeds, exponent

    def fit(self, frame, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not len(y) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Finite nonnegative aligned training targets required")
        train = attach_clean(frame)
        active = y[y != INACTIVE]
        self.active_mean_ = float(active.mean()) if len(active) else 0.0
        with threadpool_limits(limits=2):
            self.context_ = [SelectivityModel(seed=seed, **PROFILES["context_half"]).fit(train, y)
                             for seed in range(self.seeds)]
            self.kernel_ = KernelSelectivityModel(**PROFILES["kernel"]).fit(train, y)
        return self

    def predict_components(self, frame):
        query = attach_clean(frame)
        with threadpool_limits(limits=2):
            context = np.mean([model.probability(query) for model in self.context_], axis=0)
            probability = .5 * (context + self.kernel_.probability(query))
        calibrated = sharpen_probability(probability, self.exponent)
        prediction = calibrated * SENTINEL + (1 - calibrated) * self.active_mean_
        return {"hybrid_probability": probability, "probability": calibrated,
                "prediction": prediction}

    def predict(self, frame):
        return self.predict_components(frame)["prediction"]
