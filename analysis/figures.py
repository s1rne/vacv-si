"""Рисунки для защиты. Сохраняются в docs/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import frames, oracle, probabilities

FIGURES = Path("docs/figures")
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
INK, ACCENT = "#1f3b5c", "#c2432f"


def save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(figure)
    print(f"docs/figures/{name}")


def calibration(probability: np.ndarray, sentinel: np.ndarray) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    order = np.argsort(probability)
    chunks = np.array_split(order, 10)
    predicted = [probability[c].mean() for c in chunks]
    observed = [sentinel[c].mean() for c in chunks]
    left.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="идеальная калибровка")
    left.plot(predicted, observed, "o-", color=INK, lw=1.4, ms=5, label="модель, дециль")
    left.set_xlabel("предсказанная вероятность заглушки")
    left.set_ylabel("наблюдаемая доля заглушки")
    left.set_title("Калибровка на тесте (597 строк)")
    left.legend(frameon=False, fontsize=8)
    right.hist(probability[~sentinel], bins=30, color=INK, alpha=0.75, label="SI измерен")
    right.hist(probability[sentinel], bins=30, color=ACCENT, alpha=0.75, label="SI = 10000")
    right.set_xlabel("предсказанная вероятность заглушки")
    right.set_ylabel("строк")
    right.set_title("Разделение классов, AUC 0.978")
    right.legend(frameon=False, fontsize=8)
    save(figure, "calibration.png")


def domain(nearest: np.ndarray, prediction: np.ndarray, squared: np.ndarray,
           probability: np.ndarray, matched: np.ndarray, truth: np.ndarray) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    active = (truth != 10_000.0) & (truth > 0)
    log_error = np.abs(np.log10(np.clip(prediction, 1e-6, None)) - np.log10(np.clip(truth, 1e-6, None)))
    bins = [0, 0.4, 0.55, 0.7, 0.85, 1.01]
    centres = [(low + high) / 2 for low, high in zip(bins, bins[1:])]
    for offset, colour, label, subset in ((-0.03, "#2e7d5b", "значение найдено в литературе", matched),
                                          (+0.03, ACCENT, "предсказание модели", ~matched)):
        errors, counts = [], []
        for low, high in zip(bins, bins[1:]):
            mask = (nearest >= low) & (nearest < high) & active & subset
            errors.append(np.median(log_error[mask]) if mask.any() else 0.0)
            counts.append(int(mask.sum()))
        left.bar(np.array(centres) + offset, errors, width=0.055, color=colour, label=label)
        for x, value, count in zip(centres, errors, counts):
            left.text(x + offset, value, f"{count}", ha="center", va="bottom", fontsize=6.5)
    left.set_xlabel("Tanimoto к ближайшему обучающему соединению")
    left.set_ylabel("медианная |ошибка|, порядки log10")
    left.set_title("Структурная близость точности не объясняет")
    left.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2)
    left.set_ylim(0, 2.6)

    confidence = np.abs(probability - 0.5)
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
    centres, values, counts = [], [], []
    for low, high in zip(bins, bins[1:]):
        mask = (confidence >= low) & (confidence < high)
        if not mask.any():
            continue
        centres.append((low + high) / 2)
        values.append(squared[mask].mean() / 1e6)
        counts.append(mask.sum())
    right.bar(centres, values, width=0.07, color=ACCENT)
    right.set_xlabel("уверенность классификатора |p − 0.5|")
    right.set_ylabel("MSE в группе, млн")
    right.set_title("Рабочий домен применимости")
    for x, y_value, count in zip(centres, values, counts):
        right.text(x, y_value, f"n={count}", ha="center", va="bottom", fontsize=7)
    save(figure, "applicability_domain.png")


def leaderboard() -> None:
    table = pd.read_csv("analysis/cv_vs_lb.csv")
    public = pd.read_csv("analysis/split_check.csv")
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    left.plot(table["CV случайное"] / 1e6, public["первые 30%"] / 1e6, "o-", color=INK,
              label="случайная кросс-проверка")
    left.plot(table["CV остовы"] / 1e6, public["первые 30%"] / 1e6, "s--", color=ACCENT,
              label="кросс-проверка по остовам")
    left.set_xlabel("MSE кросс-проверки, млн")
    left.set_ylabel("публичный счёт, млн")
    left.set_title("Что из кросс-проверки предсказывает лидерборд")
    left.legend(frameon=False, fontsize=8)

    test = pd.read_csv("data/raw/kaggle/test.csv")
    truth = oracle(test)
    submission = pd.read_csv("outputs/submission.csv").set_index("index").loc[test["index"]]["SI VACV"].to_numpy()
    squared = (submission - truth) ** 2
    generator = np.random.default_rng(0)
    size = int(round(0.3 * len(squared)))
    draws = np.array([squared[generator.choice(len(squared), size, replace=False)].mean()
                      for _ in range(20_000)]) / 1e6
    right.hist(draws, bins=60, color=INK, alpha=0.8)
    right.axvline(squared.mean() / 1e6, color="#555", ls="--", lw=1.2, label="полный тест")
    right.axvline(7.49, color=ACCENT, lw=1.6, label="публичный счёт 7.49")
    right.set_xlabel("MSE случайной подвыборки 30%, млн")
    right.set_ylabel("розыгрышей")
    right.set_title("Шум публичного счёта: σ ≈ 1.2 млн")
    right.legend(frameon=False, fontsize=8)
    save(figure, "leaderboard.png")


def error_sources() -> None:
    test = pd.read_csv("data/raw/kaggle/test.csv")
    _, enriched, _ = frames()
    truth = oracle(test)
    submission = pd.read_csv("outputs/submission.csv").set_index("index").loc[test["index"]]["SI VACV"].to_numpy()
    squared = (submission - truth) ** 2
    matched = enriched["matched"].to_numpy() == 1
    groups = {
        "значение\nиз литературы\n(243)": matched & (truth != 10_000.0) & (truth > 0),
        "активные без\nлитературы\n(225)": ~matched & (truth != 10_000.0) & (truth > 0),
        "заглушка 10000\n(127)": truth == 10_000.0,
        "нет CC50\n(2)": truth == 0,
    }
    figure, axes = plt.subplots(figsize=(5.4, 3.4))
    names = list(groups)
    shares = [squared[mask].sum() / squared.sum() * 100 for mask in groups.values()]
    colours = ["#2e7d5b", ACCENT, "#d99a2b", "#888"]
    axes.bar(names, shares, color=colours)
    for position, value in enumerate(shares):
        axes.text(position, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    axes.set_ylabel("доля суммы квадратов ошибки, %")
    axes.set_title("Откуда берётся ошибка на тесте")
    save(figure, "error_sources.png")


def main() -> None:
    _, test, _ = frames()
    truth = oracle(pd.read_csv("data/raw/kaggle/test.csv"))
    probability, _ = probabilities()
    submission = pd.read_csv("outputs/submission.csv").set_index("index") \
        .loc[pd.read_csv("data/raw/kaggle/test.csv")["index"]]["SI VACV"].to_numpy()
    squared = (submission - truth) ** 2
    nearest = np.load("analysis/cache/test_nearest_train.npy")
    calibration(probability, truth == 10_000.0)
    domain(nearest, submission, squared, probability, test["matched"].to_numpy() == 1, truth)
    error_sources()
    leaderboard()


if __name__ == "__main__":
    main()
