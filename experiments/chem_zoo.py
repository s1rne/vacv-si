"""Набор моделей для структурного блока: классификаторы неактивности и регрессоры SI.

Состав взят из docs/research.md, раздел 2.6. Деревья и бустинги работают на сырой
конкатенации, ядровые методы и KNN — на нормированных или на счётных отпечатках,
поэтому масштабирование убрано в пайплайн каждой модели, а не сделано один раз снаружи.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

SEED = 0


def tanimoto_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Расширение коэффициента Танимото на счётные векторы: <a,b> / (|a|² + |b|² − <a,b>)."""
    dot = x @ y.T
    norm_x = np.einsum("ij,ij->i", x, x)[:, None]
    norm_y = np.einsum("ij,ij->i", y, y)[None, :]
    denominator = norm_x + norm_y - dot
    return np.divide(dot, denominator, out=np.zeros_like(dot, dtype=float), where=denominator > 0)


def tanimoto_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(a @ b)
    denominator = float(a @ a) + float(b @ b) - dot
    return 1.0 - dot / denominator if denominator > 0 else 1.0


def _scaled(estimator) -> object:
    """Отсев постоянных столбцов перед стандартизацией: без него StandardScaler даёт NaN."""
    return make_pipeline(VarianceThreshold(0.0), StandardScaler(), estimator)


class TabPFN(BaseEstimator, ClassifierMixin):
    """TabPFN v2 принимает не больше 500 столбцов, поэтому берутся самые изменчивые."""

    def __init__(self, limit: int = 300):
        self.limit = limit

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabPFN":
        from tabpfn import TabPFNClassifier

        self.columns_ = np.argsort(x.var(axis=0))[::-1][:self.limit]
        self.classes_ = np.unique(y)
        self.model_ = TabPFNClassifier(random_state=SEED).fit(x[:, self.columns_], y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(x[:, self.columns_])


CLASSIFIERS: dict[str, Callable[[], object]] = {
    "extra": lambda: ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    "rf": lambda: RandomForestClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    "lgbm": lambda: LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                                   colsample_bytree=0.3, reg_lambda=5.0, verbose=-1, random_state=SEED),
    "xgb": lambda: XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.8,
                                 colsample_bytree=0.3, reg_lambda=5.0, random_state=SEED, n_jobs=-1),
    "cat": lambda: CatBoostClassifier(iterations=600, learning_rate=0.03, depth=4, random_strength=2.0,
                                      verbose=0, allow_writing_files=False, random_seed=SEED),
    "svc_rbf": lambda: _scaled(SVC(C=3.0, probability=True, random_state=SEED)),
    "svc_tanimoto": lambda: SVC(C=3.0, kernel=tanimoto_kernel, probability=True, random_state=SEED),
    "knn_tanimoto": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance", metric=tanimoto_distance),
    "logreg": lambda: _scaled(LogisticRegression(C=0.05, max_iter=3000)),
    "tabpfn": TabPFN,
}

# Ядро Танимото и KNN определены для счётных отпечатков: на смеси с дескрипторами
# нормировка по длине вектора теряет смысл.
FINGERPRINT_ONLY = ("svc_tanimoto", "knn_tanimoto")

REGRESSORS: dict[str, Callable[[], object]] = {
    "extra": lambda: ExtraTreesRegressor(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    "rf": lambda: RandomForestRegressor(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    "lgbm": lambda: LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                                  colsample_bytree=0.3, reg_lambda=5.0, verbose=-1, random_state=SEED),
    "xgb": lambda: XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.8,
                                colsample_bytree=0.3, reg_lambda=5.0, random_state=SEED, n_jobs=-1),
    "cat": lambda: CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, random_strength=2.0,
                                     verbose=0, allow_writing_files=False, random_seed=SEED),
    "svr_rbf": lambda: _scaled(SVR(C=3.0)),
    "svr_tanimoto": lambda: SVR(C=3.0, kernel=tanimoto_kernel),
    "knn_tanimoto": lambda: KNeighborsRegressor(n_neighbors=7, weights="distance", metric=tanimoto_distance),
    "ridge": lambda: _scaled(Ridge(alpha=10.0)),
}
