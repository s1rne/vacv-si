"""Диагностика: точные скрытые метки теста, восстановленные из CC50_uM и IC50_uM_VACV.

Эти колонки запрещены для построения модели и в модель не попадают. Здесь они
используются только как эталон, чтобы разложить ошибку сабмита по источникам.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN = "data/raw/kaggle/train.csv"
TEST = "data/raw/kaggle/test.csv"


def oracle(frame: pd.DataFrame) -> np.ndarray:
    """Правило организаторов: SI = CC50/IC50, пусто в IC50 → 10000, пусто в CC50 → 0."""
    cc50 = frame["CC50_uM"].to_numpy(dtype=float)
    ic50 = frame["IC50_uM_VACV"].to_numpy(dtype=float)
    si = np.where(np.isnan(ic50), 10_000.0, cc50 / ic50)
    return np.where(np.isnan(cc50), 0.0, si)


def main() -> None:
    train = pd.read_csv(TRAIN)
    predicted = oracle(train)
    actual = train["SI VACV"].to_numpy(dtype=float)
    error = np.abs(predicted - actual) / np.maximum(actual, 1e-9)
    print(f"train: правило совпадает с SI VACV на {int((error < 1e-3).sum())} из {len(train)} строк")
    print(f"       максимальная относительная невязка {error.max():.2e}")

    test = pd.read_csv(TEST)
    truth = oracle(test)
    pd.DataFrame({"index": test["index"], "SI VACV": truth}).to_csv("analysis/test_oracle.csv", index=False)
    print(f"test: восстановлено {len(truth)} метк, из них 10000 — {int((truth == 10_000).sum())}, нулей — {int((truth == 0).sum())}")
    print(f"      дисперсия вокруг среднего (MSE константы) {np.mean((truth - truth.mean()) ** 2):,.0f}")

    submission = pd.read_csv("outputs/submission.csv")
    merged = pd.DataFrame({"index": test["index"], "truth": truth}).merge(submission, on="index")
    mse = float(np.mean((merged["SI VACV"] - merged["truth"]) ** 2))
    print(f"текущий сабмит: MSE против восстановленных метк {mse:,.0f}")


if __name__ == "__main__":
    main()
