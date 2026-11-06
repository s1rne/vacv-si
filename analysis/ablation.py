"""Вклад групп признаков, измеренный исключением, а не важностью внутри модели.

Перестановочная важность на обучающей выборке здесь бесполезна: ExtraTrees на 2215
признаках запоминает обучение, AUC на нём равен единице, и перестановка одного
столбца почти ничего не меняет. Поэтому группы признаков выключаются и модель
переобучается с честной кросс-проверкой.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from common import frames, oracle
from vacv.literature import apply
from vacv.pipeline import SelectivityModel, group_rate, open_data_matrix, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


class Ablated(SelectivityModel):
    """Та же модель с выключенной группой признаков."""

    def __init__(self, seed: int = 0, *, drop: str = "") -> None:
        super().__init__(seed)
        self.drop = drop

    def _grouped_features(self, frame: pd.DataFrame, matrix: np.ndarray) -> np.ndarray:
        rate = group_rate(self.train_frame_, self.labels_, frame)[:, None]
        open_data = open_data_matrix(frame)
        if self.drop == "структура":
            matrix = np.zeros((len(frame), 0))
        if self.drop == "кодирование группы":
            rate = np.zeros((len(frame), 0))
        if self.drop == "открытые базы":
            open_data = np.zeros((len(frame), 0))
        return np.hstack([matrix, rate, open_data])


def evaluate(drop: str, train: pd.DataFrame, y: np.ndarray, test: pd.DataFrame,
             truth: np.ndarray, splits: dict) -> dict:
    row = {"выключено": drop or "ничего"}
    labels = (y == 10_000.0).astype(int)
    for name, folds in splits.items():
        prediction = np.zeros_like(y)
        for fold in range(len(folds)):
            fit_index, check_index = fold_indices(folds, fold)
            prediction[check_index] = Ablated(drop=drop).fit(train.iloc[fit_index], y[fit_index]) \
                .predict(train.iloc[check_index])
        combined = apply(prediction, train)
        row[f"CV {name}"] = float(np.mean((combined - y) ** 2))
        row[f"AUC {name}"] = roc_auc_score(labels, prediction)
    prediction = np.mean([Ablated(seed, drop=drop).fit(train, y).predict(test) for seed in range(3)], axis=0)
    row["тест целиком"] = float(np.mean((apply(prediction, test) - truth) ** 2))
    return row


def main() -> None:
    train, test, y = frames()
    truth = oracle(pd.read_csv("data/raw/kaggle/test.csv"))
    splits = {
        "случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
        "остовы": scaffold_folds(scaffolds_of(train), n_splits=5),
    }
    rows = [evaluate(drop, train, y, test, truth, splits)
            for drop in ("", "структура", "кодирование группы", "открытые базы")]
    for row in rows:
        print(row, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv("analysis/ablation.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


if __name__ == "__main__":
    main()
