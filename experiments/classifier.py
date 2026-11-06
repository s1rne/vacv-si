"""Подбор классификатора неактивных: именно он определяет MSE соревнования."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from vacv.dataset import load
from vacv.features import descriptors, maccs, morgan_counts, parse, scaffold
from vacv.pipeline import SENTINEL
from vacv.splits import fold_indices, scaffold_folds

import pandas as pd


def blocks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    desc, morgan, keys, scaffolds = [], [], [], []
    for smiles in frame["SMILES"]:
        mol = parse(smiles)
        desc.append(descriptors(mol))
        morgan.append(morgan_counts(mol, radius=2, size=2048))
        keys.append(maccs(mol))
        scaffolds.append(scaffold(mol))
    context = pd.get_dummies(frame[["Culture"]].astype("category"), dummy_na=False)
    return {"desc": np.vstack(desc), "morgan": np.vstack(morgan), "maccs": np.vstack(keys),
            "ctx": context.to_numpy(float)}, scaffolds


MODELS = {
    "lgbm": lambda: LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                                   colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=5.0, verbose=-1),
    "rf": lambda: RandomForestClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=0),
    "extra": lambda: ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=0),
    "svm": lambda: make_pipeline(StandardScaler(), SVC(C=3.0, probability=True, random_state=0)),
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)),
}

SETS = {
    "desc+morgan+maccs+ctx": ("desc", "morgan", "maccs", "ctx"),
    "desc+maccs+ctx": ("desc", "maccs", "ctx"),
    "morgan+maccs": ("morgan", "maccs"),
    "desc+ctx": ("desc", "ctx"),
}


def score(x, y, folds, build) -> tuple[float, float]:
    inactive = (y == SENTINEL).astype(int)
    probability = np.zeros(len(y))
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        model = build().fit(x[train_index], inactive[train_index])
        probability[validation_index] = model.predict_proba(x[validation_index])[:, 1]
    active_mean = y[inactive == 0].mean()
    prediction = probability * SENTINEL + (1 - probability) * active_mean
    return roc_auc_score(inactive, probability), float(np.mean((prediction - y) ** 2))


def main() -> None:
    data = load()
    parts, scaffolds = blocks(data.train)
    y = data.y
    splits = {"scaffold": scaffold_folds(scaffolds, n_splits=5),
              "random": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)]}

    print(f"{'признаки':24} {'модель':8} {'AUC sc':>7} {'MSE sc':>12} {'AUC rn':>7} {'MSE rn':>12}")
    for set_name, keys in SETS.items():
        x = np.hstack([parts[k] for k in keys])
        for model_name, build in MODELS.items():
            results = [score(x, y, folds, build) for folds in splits.values()]
            print(f"{set_name:24} {model_name:8} {results[0][0]:7.3f} {results[0][1]:12,.0f} "
                  f"{results[1][0]:7.3f} {results[1][1]:12,.0f}")


if __name__ == "__main__":
    main()
