"""Различение соединений внутри одной публикации.

Группа strain+Culture задаёт базовую долю неактивных и покрывает 98.5% тестовых строк,
но крупные группы смешанные. Остаток неопределённости может снять только структура.
Проверяются три способа соединить групповой уровень со структурой.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.pipeline import INACTIVE, SENTINEL, ContextEncoder, structure_matrix, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds

GROUP = ["strain", "Culture"]


def forest(seed=0):
    return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)


def target_encode(train_frame, train_labels, frame, smoothing: float = 5.0) -> np.ndarray:
    """Доля неактивных в группе, сглаженная к общему среднему."""
    prior = train_labels.mean()
    stats = pd.DataFrame({"g": list(map(tuple, train_frame[GROUP].values)), "y": train_labels}) \
        .groupby("g")["y"].agg(["size", "mean"])
    encoded = (stats["size"] * stats["mean"] + smoothing * prior) / (stats["size"] + smoothing)
    keys = list(map(tuple, frame[GROUP].values))
    return np.array([encoded.get(key, prior) for key in keys])


def main() -> None:
    data = load()
    y = data.y
    labels = (y >= INACTIVE).astype(int)
    matrix = structure_matrix(data.train)
    active_mean = y[labels == 0].mean()
    splits = {"остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5),
              "случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)]}

    def mse(probability):
        return float(np.mean((probability * SENTINEL + (1 - probability) * active_mean - y) ** 2))

    for split_name, folds in splits.items():
        methods = {name: np.zeros(len(y)) for name in
                   ("смесь 0.2/0.8", "группа как признак", "стэкинг", "только группа")}
        for fold in range(len(folds)):
            train_index, validation_index = fold_indices(folds, fold)
            train_frame, valid_frame = data.train.iloc[train_index], data.train.iloc[validation_index]
            train_labels = labels[train_index]

            structure_model = forest().fit(matrix[train_index], train_labels)
            structure_valid = structure_model.predict_proba(matrix[validation_index])[:, 1]
            encoder = ContextEncoder().fit(train_frame)
            context_model = forest().fit(encoder.transform(train_frame), train_labels)
            context_valid = context_model.predict_proba(encoder.transform(valid_frame))[:, 1]
            methods["смесь 0.2/0.8"][validation_index] = 0.2 * structure_valid + 0.8 * context_valid

            rate_train = target_encode(train_frame, train_labels, train_frame)
            rate_valid = target_encode(train_frame, train_labels, valid_frame)
            methods["только группа"][validation_index] = rate_valid

            combined_train = np.hstack([matrix[train_index], rate_train[:, None]])
            combined_valid = np.hstack([matrix[validation_index], rate_valid[:, None]])
            booster = LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                                     colsample_bytree=0.3, reg_lambda=5.0, verbose=-1, random_state=0)
            methods["группа как признак"][validation_index] = booster.fit(combined_train, train_labels) \
                .predict_proba(combined_valid)[:, 1]

            inner = list(KFold(4, shuffle=True, random_state=1).split(train_index))
            meta_features = np.zeros((len(train_index), 2))
            for inner_train, inner_valid in inner:
                s = forest().fit(matrix[train_index][inner_train], train_labels[inner_train]) \
                    .predict_proba(matrix[train_index][inner_valid])[:, 1]
                meta_features[inner_valid, 0] = s
                meta_features[inner_valid, 1] = rate_train[inner_valid]
            from sklearn.linear_model import LogisticRegression
            meta = LogisticRegression(max_iter=2000).fit(meta_features, train_labels)
            methods["стэкинг"][validation_index] = meta.predict_proba(
                np.column_stack([structure_valid, rate_valid]))[:, 1]

        print(f"\n{split_name}:")
        for name, probability in methods.items():
            print(f"  {name:22} AUC {roc_auc_score(labels, probability):.3f}  MSE {mse(probability):12,.0f}")


if __name__ == "__main__":
    main()
