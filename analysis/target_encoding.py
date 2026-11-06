"""Кодирование группы: вклад строки в собственную оценку.

group_rate считает долю неактивных в группе strain+Culture по обучающим меткам. На
обучении строка попадает в свою же группу, поэтому её метка входит в собственный
признак с весом 1/(n+сглаживание). Здесь это сравнивается с кодированием «без себя»
(leave-one-out) — честным аналогом того, что видит тестовая строка.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import frames, oracle
from vacv.literature import apply
from vacv.pipeline import GROUP_COLUMNS, SMOOTHING, SelectivityModel, group_rate, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def leave_one_out_rate(frame: pd.DataFrame, labels: np.ndarray, smoothing: float = SMOOTHING) -> np.ndarray:
    """Доля неактивных в группе без вклада самой строки."""
    key = pd.Series(list(map(tuple, frame[GROUP_COLUMNS].values)))
    prior = labels.mean()
    size = key.map(key.value_counts()).to_numpy(float)
    total = key.map(pd.Series(labels).groupby(key).sum()).to_numpy(float)
    return (total - labels + smoothing * prior) / (size - 1 + smoothing)


class HonestModel(SelectivityModel):
    """Та же модель, но на обучении признак группы считается без вклада самой строки."""

    def _grouped_features(self, frame: pd.DataFrame, matrix: np.ndarray) -> np.ndarray:
        from vacv.pipeline import open_data_matrix

        if frame is self.train_frame_:
            rate = leave_one_out_rate(frame, self.labels_)[:, None]
        else:
            rate = group_rate(self.train_frame_, self.labels_, frame)[:, None]
        return np.hstack([matrix, rate, open_data_matrix(frame)])


def cross_validate(cls, frame: pd.DataFrame, y: np.ndarray, folds) -> tuple[float, float]:
    prediction = np.zeros_like(y)
    for fold in range(len(folds)):
        fit_index, check_index = fold_indices(folds, fold)
        model = cls().fit(frame.iloc[fit_index], y[fit_index])
        prediction[check_index] = model.predict(frame.iloc[check_index])
    return float(np.mean((prediction - y) ** 2)), float(np.mean((apply(prediction, frame) - y) ** 2))


def main() -> None:
    train, test, y = frames()
    truth = oracle(pd.read_csv("data/raw/kaggle/test.csv"))
    labels = (y == 10_000.0).astype(int)

    inside = group_rate(train, labels, train)
    outside = leave_one_out_rate(train, labels)
    print(f"признак группы: корреляция «со собой» и «без себя» {np.corrcoef(inside, outside)[0, 1]:.3f}")
    print(f"средний сдвиг для неактивных {np.mean(inside[labels == 1] - outside[labels == 1]):+.3f}, "
          f"для активных {np.mean(inside[labels == 0] - outside[labels == 0]):+.3f}")
    from sklearn.metrics import roc_auc_score
    print(f"AUC признака: со собой {roc_auc_score(labels, inside):.3f}, "
          f"без себя {roc_auc_score(labels, outside):.3f}")

    splits = {
        "случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
        "остовы": scaffold_folds(scaffolds_of(train), n_splits=5),
    }
    print(f"\n{'разбиение':11} {'вариант':16} {'MSE модели':>13} {'MSE + литература':>18}")
    for name, folds in splits.items():
        for label, cls in (("как сейчас", SelectivityModel), ("без себя", HonestModel)):
            model_mse, combined = cross_validate(cls, train, y, folds)
            print(f"{name:11} {label:16} {model_mse:13,.0f} {combined:18,.0f}")

    print(f"\n{'вариант':16} {'MSE на полном тесте':>21}")
    for label, cls in (("как сейчас", SelectivityModel), ("без себя", HonestModel)):
        prediction = np.mean([cls(seed).fit(train, y).predict(test) for seed in range(7)], axis=0)
        print(f"{label:16} {np.mean((apply(prediction, test) - truth) ** 2):21,.0f}")


if __name__ == "__main__":
    main()
