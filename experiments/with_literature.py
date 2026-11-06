"""Сколько даёт подстановка значений SI из открытых баз.

Оценка честная: модель обучается на фолдах, литературные значения подставляются только
на контрольной части, как это будет на тестовой выборке.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.literature import apply, attach
from vacv.pipeline import SelectivityModel, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def main() -> None:
    data = load()
    frame = attach(data.train)
    y = data.y
    splits = {"случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
              "остовы": scaffold_folds(scaffolds_of(data.train), n_splits=5)}

    for name, folds in splits.items():
        model_prediction = np.zeros_like(y)
        for fold in range(len(folds)):
            train_index, validation_index = fold_indices(folds, fold)
            model = SelectivityModel().fit(data.train.iloc[train_index], y[train_index])
            model_prediction[validation_index] = model.predict(data.train.iloc[validation_index])
        combined = apply(model_prediction, frame)
        found = frame["literature_si"].notna().to_numpy()
        print(f"\n{name}: совпадений {found.sum()} из {len(y)}")
        print(f"  только модель            {np.mean((model_prediction - y) ** 2):12,.0f}")
        print(f"  модель + литература      {np.mean((combined - y) ** 2):12,.0f}")
        print(f"  на найденных: модель {np.mean((model_prediction[found] - y[found]) ** 2):11,.0f}"
              f"  литература {np.mean((combined[found] - y[found]) ** 2):11,.0f}")
        print(f"  на ненайденных: модель   {np.mean((model_prediction[~found] - y[~found]) ** 2):12,.0f}")


if __name__ == "__main__":
    main()
