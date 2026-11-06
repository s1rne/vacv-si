"""Read-across: насколько далеко уезжает прогноз на одних ближайших соседях.

Три варианта: голос пяти соседей без модели, ExtraTrees на одних read-across-признаках,
ExtraTrees на химических блоках вместе с ними. Отпечаток для сходства выбирается тоже
здесь: ECFP радиуса 2, Avalon и их объединение дают разную окрестность.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.chem_common import FOLDS, LABELS, Y, block, matrix
from experiments.chem_zoo import CLASSIFIERS
from vacv.chemistry import RECOMMENDED
from vacv.evaluate import competition_mse, out_of_fold
from vacv.readacross import features

SPLITS = ("random", "scaffold")
SPACES = {"ecfp2": ("ecfp2",), "avalon": ("avalon",), "ecfp2+avalon": ("ecfp2", "avalon")}


def line(label: str, probability: dict[str, np.ndarray]) -> None:
    scores = []
    for split in SPLITS:
        scores += [roc_auc_score(LABELS, probability[split]),
                   competition_mse(probability[split], Y, FOLDS[split])]
    print(f"{label:40} {scores[0]:7.3f} {scores[1]:12,.0f} {scores[2]:7.3f} {scores[3]:12,.0f}")


def run(fingerprint: np.ndarray, chemistry: np.ndarray | None, model: str | None) -> dict[str, np.ndarray]:
    def predict(train_index: np.ndarray, validation_index: np.ndarray) -> np.ndarray:
        train = features(fingerprint[train_index], fingerprint[train_index], LABELS[train_index])
        validation = features(fingerprint[validation_index], fingerprint[train_index],
                              LABELS[train_index])
        if model is None:
            return validation[:, 5]  # голос соседей как готовая вероятность
        if chemistry is not None:
            train = np.hstack([chemistry[train_index], train])
            validation = np.hstack([chemistry[validation_index], validation])
        estimator = CLASSIFIERS[model]().fit(train, LABELS[train_index])
        return estimator.predict_proba(validation)[:, 1]

    return {split: out_of_fold(predict, LABELS, FOLDS[split]) for split in SPLITS}


def main() -> None:
    chemistry = matrix(RECOMMENDED)
    print(f"{'вариант':40} {'AUC rn':>7} {'MSE rn':>12} {'AUC sc':>7} {'MSE sc':>12}")
    for name, blocks in SPACES.items():
        fingerprint = np.hstack([block(key) for key in blocks])
        line(f"голос 5 соседей, {name}", run(fingerprint, None, None))
        line(f"extra на read-across, {name}", run(fingerprint, None, "extra"))
        line(f"extra, химия + read-across, {name}", run(fingerprint, chemistry, "extra"))


if __name__ == "__main__":
    main()
