"""Донастройка: вес смеси, текстовые признаки штамма, точные совпадения молекул."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.pipeline import SENTINEL, structure_matrix, title_matrix, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def forest(seed: int = 0) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)


def context_blocks(frame: pd.DataFrame, fitted: dict | None):
    text = (frame["strain"].astype(str) + " | " + frame["Culture"].astype(str)).str.lower()
    if fitted is None:
        vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"[a-zа-я0-9]+", min_df=2, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(text)
        fitted = {"vectorizer": vectorizer}
    else:
        matrix = fitted["vectorizer"].transform(text)
    return np.hstack([matrix.toarray(), title_matrix(frame["Title"])]), fitted


def dummies_blocks(frame: pd.DataFrame, fitted: dict | None):
    parts = []
    columns = {} if fitted is None else fitted["columns"]
    for name in ("strain", "Culture"):
        dummy = pd.get_dummies(frame[name].astype("category"), prefix=name)
        if fitted is None:
            columns[name] = dummy.columns
        parts.append(dummy.reindex(columns=columns[name], fill_value=0).to_numpy(float))
    return np.hstack(parts + [title_matrix(frame["Title"])]), {"columns": columns}


def out_of_fold(frame, labels, folds, encoder):
    probability = np.zeros(len(labels))
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        x_train, fitted = encoder(frame.iloc[train_index], None)
        x_valid, _ = encoder(frame.iloc[validation_index], fitted)
        probability[validation_index] = forest().fit(x_train, labels[train_index]).predict_proba(x_valid)[:, 1]
    return probability


def structure_oof(matrix, labels, folds):
    probability = np.zeros(len(labels))
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        probability[validation_index] = forest().fit(matrix[train_index], labels[train_index]) \
            .predict_proba(matrix[validation_index])[:, 1]
    return probability


def main() -> None:
    data = load()
    y = data.y
    labels = (y == SENTINEL).astype(int)
    active_mean = y[labels == 0].mean()
    matrix = structure_matrix(data.train)
    splits = {"остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5),
              "случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)]}

    for split_name, folds in splits.items():
        structure = structure_oof(matrix, labels, folds)
        encoders = {"категории": dummies_blocks, "tf-idf": context_blocks}
        print(f"\n{split_name}:")
        for encoder_name, encoder in encoders.items():
            context = out_of_fold(data.train, labels, folds, encoder)
            best = min(
                ((weight, float(np.mean(((weight * structure + (1 - weight) * context) * SENTINEL
                                         + (1 - (weight * structure + (1 - weight) * context)) * active_mean - y) ** 2)))
                 for weight in np.arange(0, 1.01, 0.1)), key=lambda item: item[1])
            blended = best[0] * structure + (1 - best[0]) * context
            print(f"  {encoder_name:12} лучший вес {best[0]:.1f}  AUC {roc_auc_score(labels, blended):.3f}  MSE {best[1]:,.0f}")


if __name__ == "__main__":
    main()
