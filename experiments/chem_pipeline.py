"""Что даёт новая химия внутри текущей смеси пайплайна.

Сравниваются четыре источника вероятности неактивности:

- `structure` — прежний структурный блок пайплайна: счётный Morgan плюс MACCS, ExtraTrees;
- `chemistry` — ансамбль vacv/structure.py по блокам химических признаков;
- `context`   — штамм, культура, обозначение соединения и признаки открытых баз;
- `grouped`   — химия плюс доля неактивных в группе strain+Culture одним признаком.

Веса перебираются по сетке на симплексе. Отдельно печатается текущая конфигурация
пайплайна (0.0 / 0.3 / 0.7), чтобы выигрыш был виден прямо.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

from experiments.chem_blend import context_matrix
from experiments.chem_common import DATA, FOLDS, LABELS, Y, matrix
from vacv.chemistry import RECOMMENDED
from vacv.evaluate import competition_mse, out_of_fold
from vacv.literature import attach
from vacv.pipeline import GROUP_COLUMNS, group_rate, open_data_matrix, structure_matrix
from vacv.structure import StructureModel

SPLITS = ("random", "scaffold")
STEP = 0.1


def forest() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=0)


def booster() -> LGBMClassifier:
    return LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                          colsample_bytree=0.3, reg_lambda=5.0, verbose=-1, random_state=0)


def grouped_oof(frame: pd.DataFrame, chemistry: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    presence = open_data_matrix(attach(frame))

    def predict(train_index: np.ndarray, validation_index: np.ndarray) -> np.ndarray:
        rate_train = group_rate(frame.iloc[train_index], LABELS[train_index], frame.iloc[train_index])
        rate_validation = group_rate(frame.iloc[train_index], LABELS[train_index],
                                     frame.iloc[validation_index])
        features = np.hstack([chemistry, presence])
        train = np.hstack([features[train_index], rate_train[:, None]])
        validation = np.hstack([features[validation_index], rate_validation[:, None]])
        return booster().fit(train, LABELS[train_index]).predict_proba(validation)[:, 1]

    return out_of_fold(predict, LABELS, folds)


def chemistry_oof(frame: pd.DataFrame, folds: list[np.ndarray]) -> np.ndarray:
    def predict(train_index: np.ndarray, validation_index: np.ndarray) -> np.ndarray:
        model = StructureModel().fit(frame.iloc[train_index], Y[train_index])
        return model.probability(frame.iloc[validation_index])

    return out_of_fold(predict, LABELS, folds)


def plain_oof(x: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    def predict(train_index: np.ndarray, validation_index: np.ndarray) -> np.ndarray:
        return forest().fit(x[train_index], LABELS[train_index]).predict_proba(x[validation_index])[:, 1]

    return out_of_fold(predict, LABELS, folds)


def main() -> None:
    frame = DATA.train
    if not set(GROUP_COLUMNS) <= set(frame.columns):
        raise ValueError("нет колонок группы")
    chemistry_features = matrix(RECOMMENDED)

    old_features = structure_matrix(frame)
    sources: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        folds = FOLDS[split]
        sources.setdefault("structure", {})[split] = plain_oof(old_features, folds)
        sources.setdefault("chemistry", {})[split] = chemistry_oof(frame, folds)
        sources.setdefault("context", {})[split] = plain_oof(context_matrix(frame, folds), folds)
        sources.setdefault("grouped_old", {})[split] = grouped_oof(frame, old_features, folds)
        sources.setdefault("grouped", {})[split] = grouped_oof(frame, chemistry_features, folds)

    def report(label: str, weights: dict[str, float]) -> float:
        scores = []
        for split in SPLITS:
            probability = sum(weight * sources[name][split] for name, weight in weights.items())
            scores += [roc_auc_score(LABELS, probability), competition_mse(probability, Y, FOLDS[split])]
        print(f"{label:46} {scores[0]:7.3f} {scores[1]:12,.0f} {scores[2]:7.3f} {scores[3]:12,.0f}")
        return scores[1]

    print(f"{'конфигурация':46} {'AUC rn':>7} {'MSE rn':>12} {'AUC sc':>7} {'MSE sc':>12}")
    for name in sources:
        report(f"{name} один", {name: 1.0})
    report("пайплайн как есть: Morgan+MACCS в grouped", {"context": 0.3, "grouped_old": 0.7})
    report("те же веса, новые блоки в grouped", {"context": 0.3, "grouped": 0.7})

    grid = np.round(np.arange(0.0, 1.0 + STEP / 2, STEP), 2)
    best = None
    for chemistry_weight, context_weight in itertools.product(grid, grid):
        if chemistry_weight + context_weight > 1.0:
            continue
        weights = {"chemistry": chemistry_weight, "context": context_weight,
                   "grouped": round(1.0 - chemistry_weight - context_weight, 2)}
        probability = sum(weight * sources[name]["random"] for name, weight in weights.items())
        mse = competition_mse(probability, Y, FOLDS["random"])
        if best is None or mse < best[0]:
            best = (mse, weights)

    print()
    report("лучшая смесь по случайному разбиению", best[1])
    print("веса: " + ", ".join(f"{name} {weight}" for name, weight in best[1].items()))


if __name__ == "__main__":
    main()
