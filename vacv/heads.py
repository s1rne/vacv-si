"""Две головы по эндпоинтам и голова неактивности, обученные на внешнем корпусе.

Разложение SI = CC50 / IC50 на две задачи даёт три вещи, которых нет у прямой регрессии
отношения:

1. Данные. Прямая регрессия SI по корпусу опирается на 367 строк, где точны обе половины.
   Головы по IC50 и CC50 используют все 1308: строка `IC50 > 100` цензурирована справа,
   но информативна, а в отношении она превращается в неинформативную единицу.
2. Перенос. Колонки CC50_uM и IC50_uM_VACV файлов соревнования запрещены правилами,
   поэтому обе головы учатся только на корпусе и на выборку соревнования переносятся
   без её целевой переменной вообще.
3. Семантика заглушки. SI = 10000 в соревновании означает «IC50 отсутствует, соединение
   против VACV неактивно». В корпусе этому ровно соответствует `ic50_relation == '>'`:
   IC50 не определена, известна только нижняя граница. Голова неактивности учится на этой
   метке и переносится на заглушку.

Цензурирование обрабатывается objective `survival:aft` XGBoost: метка задаётся парой границ,
правая цензура — это [y, +inf). Альтернатива «выбросить строки со знаком» проверяется
в `experiments/endpoint_cv.py`; выбранный режим задаётся параметром `censored`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier

from .endpoints import CONTEXT_COLUMNS, competition_context
from .features import descriptors, maccs, morgan_counts, parse

ENDPOINTS = ("ic50", "cc50")

AFT_PARAMS = {
    "objective": "survival:aft",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.0,
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "verbosity": 0,
}
ROUNDS = 400

_cache: dict[str, np.ndarray] = {}


def structure_row(smiles: str) -> np.ndarray:
    """Дескрипторы RDKit, счётные отпечатки Моргана и MACCS одной молекулы."""
    cached = _cache.get(smiles)
    if cached is None:
        mol = parse(smiles)
        if mol is None:
            raise ValueError(f"не разбирается SMILES: {smiles}")
        cached = np.concatenate([descriptors(mol), morgan_counts(mol), maccs(mol)]).astype(np.float32)
        _cache[smiles] = cached
    return cached


def structure_matrix(smiles: pd.Series) -> np.ndarray:
    return np.vstack([structure_row(value) for value in smiles])


def corpus_matrix(corpus: pd.DataFrame, context: bool = True) -> np.ndarray:
    parts = [structure_matrix(corpus["smiles"])]
    if context:
        parts.append(corpus[CONTEXT_COLUMNS].to_numpy(np.float32))
    return np.hstack(parts)


def competition_matrix(frame: pd.DataFrame, context: bool = True) -> np.ndarray:
    parts = [structure_matrix(frame["SMILES"])]
    if context:
        parts.append(competition_context(frame).to_numpy(np.float32))
    return np.hstack(parts)


def inactivity_classifier(seed: int = 0) -> LGBMClassifier:
    """Голова «IC50 не определена»: метка корпуса `ic50_relation == '>'`."""
    return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                          colsample_bytree=0.3, reg_lambda=5.0, verbose=-1, random_state=seed)


def _aft(matrix: np.ndarray, lower: np.ndarray, upper: np.ndarray, seed: int) -> xgb.Booster:
    data = xgb.DMatrix(matrix)
    data.set_float_info("label_lower_bound", lower)
    data.set_float_info("label_upper_bound", upper)
    return xgb.train({**AFT_PARAMS, "seed": seed}, data, num_boost_round=ROUNDS)


class EndpointHeads:
    """IC50, CC50 и вероятность неактивности по строению и тест-системе.

    `censored=False` обучает регрессии только на точных значениях: это и есть сравнение
    с наивным выбрасыванием строк со знаком «>». `context=False` убирает линию клеток
    и штамм, то есть мерит, сколько даёт знание тест-системы.
    """

    def __init__(self, censored: bool = True, context: bool = True, seed: int = 0):
        self.censored = censored
        self.context = context
        self.seed = seed

    def fit(self, corpus: pd.DataFrame) -> "EndpointHeads":
        matrix = corpus_matrix(corpus, self.context)
        self.boosters_ = {}
        self.scale_ = {}
        for endpoint in ENDPOINTS:
            lower = corpus[f"{endpoint}_lower"].to_numpy(float)
            upper = corpus[f"{endpoint}_upper"].to_numpy(float)
            keep = np.isfinite(upper) if not self.censored else np.ones(len(corpus), bool)
            self.boosters_[endpoint] = _aft(matrix[keep], lower[keep], upper[keep], self.seed)
            self.scale_[endpoint] = float(np.log10(corpus.loc[keep, endpoint]).std())
        self.classifier_ = inactivity_classifier(self.seed).fit(matrix, corpus["inactive"].to_numpy(int))
        self.prior_ = float(corpus["inactive"].mean())
        return self

    def _predict_matrix(self, matrix: np.ndarray) -> pd.DataFrame:
        out = {}
        for endpoint in ENDPOINTS:
            out[f"log_{endpoint}"] = np.log10(self.boosters_[endpoint].predict(xgb.DMatrix(matrix)))
        out["log_si"] = out["log_cc50"] - out["log_ic50"]
        out["p_inactive"] = self.classifier_.predict_proba(matrix)[:, 1]
        return pd.DataFrame(out)

    def predict_corpus(self, corpus: pd.DataFrame) -> pd.DataFrame:
        return self._predict_matrix(corpus_matrix(corpus, self.context))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Головы на выборке соревнования: целевая переменная соревнования не нужна."""
        return self._predict_matrix(competition_matrix(frame, self.context))


HEAD_FEATURES = ["log_ic50", "log_cc50", "log_si", "p_inactive"]
