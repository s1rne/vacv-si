"""Диагностика: точные скрытые метки теста, восстановленные из CC50_uM и IC50_uM_VACV.

Эти колонки запрещены для построения модели и в модель не попадают: vacv/dataset.py
удаляет их при загрузке. Здесь они нужны как эталон проверки, по которому разложена
ошибка сабмита. Ни один параметр модели по ним не подбирался.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import oracle


def main() -> None:
    train = pd.read_csv("data/raw/kaggle/train.csv")
    predicted = oracle(train)
    actual = train["SI VACV"].to_numpy(dtype=float)
    error = np.abs(predicted - actual) / np.maximum(actual, 1e-9)
    print(f"train: правило совпадает с SI VACV на {int((error < 1e-3).sum())} из {len(train)} строк")
    print(f"       максимальная относительная невязка {error.max():.2e}")

    test = pd.read_csv("data/raw/kaggle/test.csv")
    truth = oracle(test)
    print(f"test: восстановлено {len(truth)} меток, из них 10000 — {int((truth == 10_000).sum())}, "
          f"нулей — {int((truth == 0).sum())}")
    print(f"      MSE константы (среднее) {np.mean((truth - truth.mean()) ** 2):,.0f}")

    submission = pd.read_csv("outputs/submission.csv")
    merged = pd.DataFrame({"index": test["index"], "truth": truth}).merge(submission, on="index")
    print(f"текущий сабмит: MSE против восстановленных меток "
          f"{np.mean((merged['SI VACV'] - merged['truth']) ** 2):,.0f}")


if __name__ == "__main__":
    main()
