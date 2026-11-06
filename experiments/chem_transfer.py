"""Перенос с открытых данных: помогает ли химии внешняя выборка по VACV.

В `data/processed/vacv_si.csv` 1308 соединений, собранных из ChEMBL и PubChem, и у 760
из них IC50 против VACV опубликована как «больше чем», то есть активность не обнаружена.
Это та же ситуация, что отсутствующая IC50 в данных соревнования, только метка приходит
из другой публикации и другого прогона теста.

Проверяются два способа: добавить внешние строки в обучение с весом и обучить на них
отдельную модель, прогноз которой идёт одним столбцом к химическим признакам.

Утечка закрыта грубо: из внешней выборки убираются все соединения, остов Мурко которых
встречается в контрольной части фолда. Иначе аналог из той же статьи попадёт в обучение.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.chem_common import FOLDS, LABELS, SMILES, Y, classifier_oof, matrix
from experiments.chem_zoo import CLASSIFIERS
from vacv.chemistry import RECOMMENDED, block_matrix, parse
from vacv.evaluate import competition_mse, out_of_fold
from vacv.features import scaffold
from vacv.splits import fold_indices

EXTERNAL = "data/processed/vacv_si.csv"
NO_ACTIVITY = (">", ">=")
MODEL = "extra"


def external() -> tuple[np.ndarray, np.ndarray, list[str]]:
    table = pd.read_csv(EXTERNAL, usecols=["smiles", "ic50_relation"]).dropna(subset=["smiles"])
    table = table.drop_duplicates("smiles").reset_index(drop=True)
    labels = table["ic50_relation"].isin(NO_ACTIVITY).to_numpy(int)
    scaffolds = [scaffold(parse(text)) for text in table["smiles"]]
    return matrix_of(list(table["smiles"])), labels, scaffolds


def matrix_of(smiles: list[str]) -> np.ndarray:
    return np.hstack([block_matrix(smiles, name) for name in RECOMMENDED])


def main() -> None:
    x = matrix(RECOMMENDED)
    external_x, external_labels, external_scaffolds = external()
    own_scaffolds = [scaffold(parse(text)) for text in SMILES]

    print(f"{'способ':34} {'AUC rn':>7} {'MSE rn':>12} {'AUC sc':>7} {'MSE sc':>12}")

    def allowed(validation_index: np.ndarray) -> np.ndarray:
        blocked = {own_scaffolds[i] for i in validation_index if own_scaffolds[i]}
        return np.array([key not in blocked for key in external_scaffolds])

    for weight in (0.0, 0.1, 0.25, 0.5, 1.0):
        scores = []
        for split in ("random", "scaffold"):
            folds = FOLDS[split]

            def predict(train_index: np.ndarray, validation_index: np.ndarray) -> np.ndarray:
                keep = allowed(validation_index)
                features = np.vstack([x[train_index], external_x[keep]])
                labels = np.concatenate([LABELS[train_index], external_labels[keep]])
                sample_weight = np.concatenate([np.ones(len(train_index)),
                                               np.full(keep.sum(), weight)])
                model = CLASSIFIERS[MODEL]()
                model.fit(features, labels, sample_weight=sample_weight)
                return model.predict_proba(x[validation_index])[:, 1]

            probability = (classifier_oof(MODEL, x, folds) if weight == 0.0
                           else out_of_fold(predict, LABELS, folds))
            scores += [roc_auc_score(LABELS, probability), competition_mse(probability, Y, folds)]
        label = "только соревнование" if weight == 0.0 else f"+ внешние, вес {weight}"
        print(f"{label:34} {scores[0]:7.3f} {scores[1]:12,.0f} {scores[2]:7.3f} {scores[3]:12,.0f}")

    scores = []
    for split in ("random", "scaffold"):
        folds = FOLDS[split]

        def predict(train_index: np.ndarray, validation_index: np.ndarray) -> np.ndarray:
            keep = allowed(validation_index)
            auxiliary = CLASSIFIERS[MODEL]().fit(external_x[keep], external_labels[keep])
            column = auxiliary.predict_proba(x)[:, 1][:, None]
            extended = np.hstack([x, column])
            model = CLASSIFIERS[MODEL]().fit(extended[train_index], LABELS[train_index])
            return model.predict_proba(extended[validation_index])[:, 1]

        probability = out_of_fold(predict, LABELS, folds)
        scores += [roc_auc_score(LABELS, probability), competition_mse(probability, Y, folds)]
    print(f"{'прогноз внешней модели столбцом':34} {scores[0]:7.3f} {scores[1]:12,.0f} "
          f"{scores[2]:7.3f} {scores[3]:12,.0f}")


if __name__ == "__main__":
    main()
