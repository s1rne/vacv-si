"""Проверка признаков из контекста записи: strain, Culture, Title.

Title содержит обозначение соединения из статьи-источника. Вместе со strain и Culture
он косвенно указывает на публикацию, а доля неактивных внутри одной публикации
сильно смещена. Это законный сигнал: запрещены только CC50_uM и IC50_uM_VACV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.features import maccs, morgan_counts, parse, scaffold
from vacv.pipeline import SENTINEL
from vacv.splits import fold_indices, scaffold_folds


def title_features(titles: pd.Series) -> np.ndarray:
    text = titles.astype(str)
    return np.column_stack([
        text.str.len(),
        text.str.count(r"\d"),
        text.str.match(r"^\d+$").astype(float),
        text.str.contains("-").astype(float),
        text.str.extract(r"^(\d+)", expand=False).astype(float).fillna(-1),
    ])


def build(frame: pd.DataFrame, categories: dict[str, pd.Index] | None = None):
    structure = np.vstack([np.concatenate([morgan_counts(parse(s)), maccs(parse(s))]) for s in frame["SMILES"]])
    scaffolds = [scaffold(parse(s)) for s in frame["SMILES"]]
    context = {}
    for column in ("strain", "Culture"):
        dummies = pd.get_dummies(frame[column].astype("category"), prefix=column)
        context[column] = dummies if categories is None else dummies.reindex(columns=categories[column], fill_value=0)
    return structure, scaffolds, context, title_features(frame["Title"])


def score(x, y, folds) -> tuple[float, float]:
    inactive = (y == SENTINEL).astype(int)
    probability = np.zeros(len(y))
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        model = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=0)
        model.fit(x[train_index], inactive[train_index])
        probability[validation_index] = model.predict_proba(x[validation_index])[:, 1]
    prediction = probability * SENTINEL + (1 - probability) * y[inactive == 0].mean()
    return roc_auc_score(inactive, probability), float(np.mean((prediction - y) ** 2))


def main() -> None:
    data = load()
    structure, scaffolds, context, titles = build(data.train)
    y = data.y
    splits = {"scaffold": scaffold_folds(scaffolds, n_splits=5),
              "random": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)]}

    variants = {
        "структура": [structure],
        "+ Culture": [structure, context["Culture"].to_numpy(float)],
        "+ strain": [structure, context["strain"].to_numpy(float)],
        "+ strain + Culture": [structure, context["strain"].to_numpy(float), context["Culture"].to_numpy(float)],
        "+ strain + Culture + Title": [structure, context["strain"].to_numpy(float),
                                       context["Culture"].to_numpy(float), titles],
        "только контекст": [context["strain"].to_numpy(float), context["Culture"].to_numpy(float), titles],
    }
    print(f"{'признаки':30} {'AUC sc':>7} {'MSE sc':>12} {'AUC rn':>7} {'MSE rn':>12}")
    for name, parts in variants.items():
        x = np.hstack(parts)
        results = [score(x, y, folds) for folds in splits.values()]
        print(f"{name:30} {results[0][0]:7.3f} {results[0][1]:12,.0f} {results[1][0]:7.3f} {results[1][1]:12,.0f}")


if __name__ == "__main__":
    main()
