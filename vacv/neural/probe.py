"""Головы поверх замороженных эмбеддингов.

Классификатор неактивных определяет MSE соревнования, поэтому его качество и измеряется.
Чтобы не выдавать за результат подобранный по контролю гиперпараметр, у логистической
регрессии C выбирается вложенной трёхфолдовой проверкой внутри обучающей части фолда.
MLP и ExtraTrees берутся с фиксированными настройками.

ExtraTrees на эмбеддинге нужен для разделения двух причин проигрыша: слабое представление
или слабая голова. Если лес на эмбеддинге тоже проигрывает лесу на отпечатках, дело
в представлении.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..protocol import RANDOM_SEED, Result, Task, score

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def linear(seed: int = RANDOM_SEED):
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    return make_pipeline(
        StandardScaler(),
        GridSearchCV(LogisticRegression(max_iter=5000, random_state=seed),
                     {"C": C_GRID}, scoring="roc_auc", cv=inner, n_jobs=-1))


def mlp(seed: int = RANDOM_SEED):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, learning_rate_init=1e-3,
                      max_iter=800, early_stopping=True, n_iter_no_change=30,
                      validation_fraction=0.15, random_state=seed))


def forest(seed: int = RANDOM_SEED):
    return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)


HEADS = {"linear": linear, "mlp": mlp, "extra": forest}


def out_of_fold(job: Task, split: str, x: np.ndarray, head) -> tuple[np.ndarray, float]:
    probability = np.zeros(len(job.y))
    started = time.perf_counter()
    for train_index, validation_index in job.splits(split):
        model = head().fit(x[train_index], job.labels[train_index])
        probability[validation_index] = model.predict_proba(x[validation_index])[:, 1]
    return probability, time.perf_counter() - started


def probe(job: Task, split: str, x: np.ndarray, name: str, head: str = "linear",
          extra_seconds: float = 0.0) -> Result:
    probability, elapsed = out_of_fold(job, split, x, HEADS[head])
    auc, mse = score(job.labels, job.y, probability, job.active_mean)
    return Result(f"{name} + {head}", split, auc, mse, elapsed + extra_seconds, probability)


def active_regression(job: Task, split: str, x: np.ndarray) -> np.ndarray:
    """Оценка SI активных из эмбеддинга, out-of-fold, в логарифме.

    Обучается только на активных строках обучающей части фолда: у неактивных измерения нет.
    """
    prediction = np.full(len(job.y), job.active_mean)
    for train_index, validation_index in job.splits(split):
        active = train_index[job.labels[train_index] == 0]
        target = np.log10(np.clip(job.y[active], 1e-3, None))
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))) \
            .fit(x[active], target)
        prediction[validation_index] = np.power(10.0, model.predict(x[validation_index]))
    return prediction
