"""Сравнение вариантов оценки условного среднего. Метрика соревнования: MSE на исходной шкале."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.pipeline import ContextEncoder, SENTINEL, scaffolds_of, structure_matrix
from vacv.splits import fold_indices, scaffold_folds

CLF = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10,
           colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=5.0, verbose=-1)


def lognormal(x_train, y_train, x_test, active):
    model = LGBMRegressor(**CLF, random_state=0).fit(x_train[active], np.log10(np.clip(y_train[active], 1e-3, None)))
    residual = np.log10(np.clip(y_train[active], 1e-3, None)) - model.predict(x_train[active])
    return np.power(10.0, model.predict(x_test) + 0.5 * np.log(10.0) * residual.std() ** 2)


def raw_mean(x_train, y_train, x_test, active):
    return LGBMRegressor(**CLF, random_state=0).fit(x_train[active], y_train[active]).predict(x_test)


def constant_mean(x_train, y_train, x_test, active):
    return np.full(len(x_test), y_train[active].mean())


REGRESSORS = {"lognormal": lognormal, "raw": raw_mean, "const": constant_mean}


def evaluate(x, y, folds, regressor, calibrate: bool) -> float:
    prediction = np.zeros_like(y)
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        x_train, y_train = x[train_index], y[train_index]
        inactive = y_train == SENTINEL
        base = LGBMClassifier(**CLF, random_state=0)
        classifier = CalibratedClassifierCV(base, method="isotonic", cv=3) if calibrate else base
        classifier.fit(x_train, inactive.astype(int))
        probability = classifier.predict_proba(x[validation_index])[:, 1]
        active_mean = np.clip(REGRESSORS[regressor](x_train, y_train, x[validation_index], ~inactive), 0, SENTINEL)
        prediction[validation_index] = probability * SENTINEL + (1 - probability) * active_mean
    return float(np.mean((prediction - y) ** 2))


def main() -> None:
    data = load()
    scaffolds = scaffolds_of(data.train)
    x = np.hstack([structure_matrix(data.train), ContextEncoder().fit(data.train).transform(data.train)])
    y = data.y

    splits = {
        "scaffold": scaffold_folds(scaffolds, n_splits=5),
        "random": [index for _, index in KFold(5, shuffle=True, random_state=0).split(x)],
    }
    print(f"{'вариант':28} {'scaffold':>12} {'random':>12}")
    print(f"{'константа = среднее':28} {np.mean((y.mean() - y) ** 2):12,.0f} {np.mean((y.mean() - y) ** 2):12,.0f}")
    for regressor in REGRESSORS:
        for calibrate in (False, True):
            name = f"{regressor}{' + калибровка' if calibrate else ''}"
            scores = [evaluate(x, y, folds, regressor, calibrate) for folds in splits.values()]
            print(f"{name:28} {scores[0]:12,.0f} {scores[1]:12,.0f}")


if __name__ == "__main__":
    main()
