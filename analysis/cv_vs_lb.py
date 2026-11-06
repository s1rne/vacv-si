"""Сопоставление кросс-проверки с публичным лидербордом.

Поколения модели восстановлены из истории коммитов и воспроизведены текущим кодом:
веса смеси, наличие признаков открытых баз и подстановки литературы задаются
параметрами, а не правкой файлов. Для каждого поколения считаются MSE при случайном
разбиении, при разбиении по остовам Мурко и MSE на полном тесте, восстановленном из
колонок CC50_uM и IC50_uM_VACV.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import frames, oracle
from vacv.literature import apply
from vacv.pipeline import SelectivityModel, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds

#: Публичный лидерборд последовательных сабмитов, в порядке отправки.
LEADERBOARD = [11.35e6, 9.29e6, 8.59e6, 7.56e6, 7.49e6]


@dataclass
class Generation:
    name: str
    weights: dict[str, float]
    literature: bool = False
    open_data: bool = False
    exact_sentinel: bool = False
    lb: float = field(default=float("nan"))


GENERATIONS = [
    Generation("структура+контекст", {"structure": 0.2, "context": 0.8, "grouped": 0.0}, lb=LEADERBOARD[0]),
    Generation("+ кодирование группы", {"structure": 0.0, "context": 0.3, "grouped": 0.7}, lb=LEADERBOARD[1]),
    Generation("+ литература", {"structure": 0.0, "context": 0.3, "grouped": 0.7},
               literature=True, lb=LEADERBOARD[2]),
    Generation("+ признаки присутствия", {"structure": 0.0, "context": 0.3, "grouped": 0.7},
               literature=True, open_data=True, lb=LEADERBOARD[3]),
    Generation("+ точная метка 10000", {"structure": 0.0, "context": 0.3, "grouped": 0.7},
               literature=True, open_data=True, exact_sentinel=True, lb=LEADERBOARD[4]),
]

OPEN_DATA_COLUMNS = ["matched", "nn1_tanimoto", "nn2_tanimoto", "nn3_tanimoto", "n_antiviral",
                     "has_antiviral", "n_records_skeleton", "best_ic50_um", "best_cc50_um", "n_cytotox"]


def strip(frame: pd.DataFrame, generation: Generation) -> pd.DataFrame:
    """Убирает колонки, которых у поколения ещё не было."""
    drop = [] if generation.open_data else [c for c in OPEN_DATA_COLUMNS if c in frame.columns]
    return frame.drop(columns=drop)


def labels_of(y: np.ndarray, generation: Generation) -> np.ndarray:
    return (y == 10_000.0) if generation.exact_sentinel else (y >= 9_999.0)


def fit_predict(fit_frame: pd.DataFrame, y: np.ndarray, target: pd.DataFrame,
                generation: Generation, seed: int = 0) -> np.ndarray:
    """Одна модель поколения. Метка неактивности задаётся поколением."""
    relabelled = np.where(labels_of(y, generation), 10_000.0, np.where(y == 10_000.0, 9_999.9, y))
    model = SelectivityModel(seed, weights=generation.weights)
    model.fit(strip(fit_frame, generation), relabelled)
    prediction = model.predict(strip(target, generation))
    return prediction


def cross_validate(frame: pd.DataFrame, y: np.ndarray, folds, generation: Generation) -> float:
    prediction = np.zeros_like(y)
    for fold in range(len(folds)):
        fit_index, check_index = fold_indices(folds, fold)
        prediction[check_index] = fit_predict(frame.iloc[fit_index], y[fit_index],
                                              frame.iloc[check_index], generation)
    if generation.literature:
        prediction = apply(prediction, frame)
    return float(np.mean((prediction - y) ** 2))


def main() -> None:
    train, test, y = frames()
    truth = oracle(pd.read_csv("data/raw/kaggle/test.csv"))
    splits = {
        "случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
        "остовы": scaffold_folds(scaffolds_of(train), n_splits=5),
    }

    rows = []
    for generation in GENERATIONS:
        prediction = np.mean([fit_predict(train, y, test, generation, seed) for seed in range(7)], axis=0)
        if generation.literature:
            prediction = apply(prediction, test)
        rows.append({
            "поколение": generation.name,
            "LB": generation.lb,
            "тест целиком": float(np.mean((prediction - truth) ** 2)),
            "CV случайное": cross_validate(train, y, splits["случайное"], generation),
            "CV остовы": cross_validate(train, y, splits["остовы"], generation),
        })
        print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    table.to_csv("analysis/cv_vs_lb.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    for column in ("CV случайное", "CV остовы", "тест целиком"):
        x, lb = table[column].to_numpy(), table["LB"].to_numpy()
        slope, intercept = np.polyfit(x, lb, 1)
        residual = lb - (slope * x + intercept)
        print(f"\n{column}: Spearman {pd.Series(x).corr(pd.Series(lb), method='spearman'):+.2f}, "
              f"Pearson {np.corrcoef(x, lb)[0, 1]:+.3f}")
        print(f"  LB ≈ {slope:.2f}·CV + {intercept:,.0f}, остаток RMS {np.sqrt((residual ** 2).mean()):,.0f}")

    Path("analysis/cv_vs_lb.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
