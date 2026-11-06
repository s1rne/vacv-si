"""Структурный домен отдельно для соединений с литературным значением и без него."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import frames, oracle

BINS = [0, 0.4, 0.55, 0.7, 0.85, 1.01]


def main() -> None:
    _, test, _ = frames()
    raw = pd.read_csv("data/raw/kaggle/test.csv")
    truth = oracle(raw)
    submission = pd.read_csv("outputs/submission.csv").set_index("index").loc[raw["index"]]["SI VACV"].to_numpy()
    nearest = np.load("analysis/cache/test_nearest_train.npy")
    matched = test["matched"].to_numpy() == 1
    active = (truth != 10_000.0) & (truth > 0)
    log_error = np.abs(np.log10(np.clip(submission, 1e-6, None)) - np.log10(np.clip(truth, 1e-6, None)))

    rows = []
    for low, high in zip(BINS, BINS[1:]):
        band = (nearest >= low) & (nearest < high)
        row = {"интервал": f"{low:.2f}–{high:.2f}"}
        for name, mask in (("с литературой", matched), ("без литературы", ~matched)):
            subset = band & mask & active
            row[f"{name}, n"] = int(subset.sum())
            row[f"{name}, медиана log10"] = float(np.median(log_error[subset])) if subset.any() else np.nan
        sentinel = band & ~matched
        row["доля заглушки без литературы"] = float((truth[sentinel] == 10_000.0).mean()) if sentinel.any() else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv("analysis/domain_split.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    for name, mask in (("с литературой", matched), ("без литературы", ~matched)):
        subset = mask & active
        rho = pd.Series(nearest[subset]).corr(pd.Series(log_error[subset]), method="spearman")
        print(f"\n{name}: Spearman(Tanimoto, |ошибка log10|) = {rho:+.3f} на {subset.sum()} строках")


if __name__ == "__main__":
    main()
