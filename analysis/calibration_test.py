"""Калибровка вероятности неактивности, проверенная по восстановленным меткам теста."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from common import frames, probabilities, test_truth


def main() -> None:
    train, test, y = frames()
    truth = test_truth()
    probability, matrix = probabilities()
    sentinel = (truth == 10_000).astype(int)

    print(f"средний SI активных на обучении: {y[y != 10_000].mean():.1f}")
    print(f"  без двух нулей и строки SI=36066: "
          f"{y[(y != 10_000) & (y > 0) & (y < 10_000)].mean():.1f}")
    print(f"средний SI активных на тесте (восстановленный): {truth[(truth != 10_000)].mean():.1f}")
    print(f"доля неактивных: обучение {np.mean(y == 10_000):.1%}, тест {sentinel.mean():.1%}\n")

    print(f"AUC вероятности на тесте: {roc_auc_score(sentinel, probability):.3f}")
    print(f"средняя вероятность {probability.mean():.3f} против фактической доли {sentinel.mean():.3f}")
    matched = test["matched"].to_numpy() == 1
    print(f"  на 243 строках с литературой: p={probability[matched].mean():.3f}, факт={sentinel[matched].mean():.3f}")
    print(f"  на 354 строках без литературы: p={probability[~matched].mean():.3f}, факт={sentinel[~matched].mean():.3f}")
    print(f"AUC только на строках без литературы: {roc_auc_score(sentinel[~matched], probability[~matched]):.3f}\n")

    print("калибровочная таблица (дециль вероятности):")
    order = np.argsort(probability)
    print(f"{'интервал p':>18} {'строк':>6} {'средняя p':>10} {'факт':>7}")
    for chunk in np.array_split(order, 10):
        print(f"{probability[chunk].min():7.3f}–{probability[chunk].max():<10.3f} {len(chunk):6d} "
              f"{probability[chunk].mean():10.3f} {sentinel[chunk].mean():7.3f}")

    print(f"\nразброс по сидам: средний std вероятности {matrix.std(axis=0).mean():.4f}, "
          f"максимум {matrix.std(axis=0).max():.4f}")


if __name__ == "__main__":
    main()
