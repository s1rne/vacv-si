"""Неопределённость предсказания и её проверка.

Предсказание двухчастное, поэтому неопределённость тоже двухчастная:
1) вероятность заглушки p — она и есть неопределённость классификации, её проверяют
   калибровкой, а не интервалом;
2) для активного соединения — интервал на log10 SI, построенный сплит-конформным
   методом по обучающей выборке.
Разброс по сидам отдельно измеряется, чтобы показать, что он неопределённостью не является.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import frames, oracle, probabilities
from vacv.literature import apply
from vacv.pipeline import SelectivityModel
from vacv.splits import fold_indices


def bootstrap_probability(train: pd.DataFrame, y: np.ndarray, test: pd.DataFrame,
                          draws: int = 15) -> np.ndarray:
    """Вероятности при обучении на бутстреп-выборках: неопределённость самой модели."""
    generator = np.random.default_rng(0)
    out = []
    for draw in range(draws):
        index = generator.choice(len(train), len(train), replace=True)
        if len(np.unique(y[index] == 10_000.0)) < 2:
            continue
        out.append(SelectivityModel(draw).fit(train.iloc[index], y[index]).probability(test))
    return np.vstack(out)


def main() -> None:
    train, test, y = frames()
    truth = oracle(pd.read_csv("data/raw/kaggle/test.csv"))
    probability, seeds = probabilities()
    sentinel = truth == 10_000.0

    print("1. разброс по сидам (та же выборка, другой random_state)")
    print(f"   средний std {seeds.std(axis=0).mean():.4f}, 95-й процентиль {np.percentile(seeds.std(axis=0), 95):.4f}")
    print("   это не неопределённость предсказания, а шум ансамбля; усреднение по 7 сидам его снимает")

    path = "analysis/cache/bootstrap_probability.npy"
    try:
        bootstrap = np.load(path)
    except FileNotFoundError:
        bootstrap = bootstrap_probability(train, y, test)
        np.save(path, bootstrap)
    spread = bootstrap.std(axis=0)
    print(f"\n2. бутстреп по обучающим строкам ({len(bootstrap)} выборок)")
    print(f"   средний std вероятности {spread.mean():.3f}, 95-й процентиль {np.percentile(spread, 95):.3f}")
    print(f"   {'квартиль разброса':>20} {'строк':>6} {'MSE':>13} {'|p−факт|':>10}")
    submission = pd.read_csv("outputs/submission.csv").set_index("index") \
        .loc[pd.read_csv("data/raw/kaggle/test.csv")["index"]]["SI VACV"].to_numpy()
    squared = (submission - truth) ** 2
    for name, chunk in zip(("1 (узкий)", "2", "3", "4 (широкий)"), np.array_split(np.argsort(spread), 4)):
        print(f"   {name:>20} {len(chunk):6d} {squared[chunk].mean():13,.0f} "
              f"{np.abs(probability[chunk] - sentinel[chunk]).mean():10.3f}")

    print("\n3. калибровка вероятности: ошибка Брайера и её разложение")
    brier = np.mean((probability - sentinel) ** 2)
    base = sentinel.mean()
    print(f"   Брайер {brier:.4f}; у константы {base:.3f} было бы {np.mean((base - sentinel) ** 2):.4f}")
    print(f"   Brier skill score {1 - brier / np.mean((base - sentinel) ** 2):.3f}")
    bins = np.array_split(np.argsort(probability), 10)
    reliability = np.mean([len(c) * (probability[c].mean() - sentinel[c].mean()) ** 2 for c in bins]) \
        / (len(probability) / 10)
    print(f"   составляющая надёжности (чем меньше, тем точнее калибровка) {reliability:.4f}")

    print("\n4. конформный интервал на log10 SI для активных соединений")
    folds = [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)]
    out_of_fold = np.zeros_like(y)
    for fold in range(len(folds)):
        fit_index, check_index = fold_indices(folds, fold)
        out_of_fold[check_index] = SelectivityModel().fit(train.iloc[fit_index], y[fit_index]) \
            .predict(train.iloc[check_index])
    out_of_fold = apply(out_of_fold, train)
    active = (y != 10_000.0) & (y > 0)
    residual = np.abs(np.log10(np.clip(out_of_fold[active], 1e-6, None)) - np.log10(y[active]))
    print(f"   калибровочных строк {active.sum()}")
    for level in (0.5, 0.8, 0.9, 0.95):
        width = float(np.quantile(residual, level))
        test_active = ~sentinel & (truth > 0)
        test_residual = np.abs(np.log10(np.clip(submission[test_active], 1e-6, None))
                               - np.log10(truth[test_active]))
        print(f"   уровень {level:.0%}: интервал ±{width:.2f} в log10 (×/÷{10 ** width:.1f}), "
              f"фактическое покрытие на тесте {np.mean(test_residual <= width):.1%}")

    print("\n   отдельно по тому, найдено ли значение в литературе:")
    for name, mask in (("найдено", test["matched"].to_numpy() == 1),
                       ("не найдено", test["matched"].to_numpy() == 0)):
        subset = mask & ~sentinel & (truth > 0)
        calibration_mask = (train["matched"].to_numpy() == 1) if name == "найдено" \
            else (train["matched"].to_numpy() == 0)
        subset_residual = np.abs(np.log10(np.clip(out_of_fold[active & calibration_mask], 1e-6, None))
                                 - np.log10(y[active & calibration_mask]))
        width = float(np.quantile(subset_residual, 0.9))
        actual = np.abs(np.log10(np.clip(submission[subset], 1e-6, None)) - np.log10(truth[subset]))
        print(f"   {name:12} калибровка на {active.sum() and (active & calibration_mask).sum():3d} строках, "
              f"интервал 90% = ±{width:.2f} (×/÷{10 ** width:.0f}), покрытие {np.mean(actual <= width):.1%}")


if __name__ == "__main__":
    main()
