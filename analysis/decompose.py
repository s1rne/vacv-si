"""Разложение ошибки текущего сабмита по строкам теста."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import frames, oracle


def main() -> None:
    test = pd.read_csv("data/raw/kaggle/test.csv")
    truth = oracle(test)
    sub = pd.read_csv("outputs/submission.csv").set_index("index").loc[test["index"]]["SI VACV"].to_numpy()

    _, frame, _ = frames()
    sq = (sub - truth) ** 2
    total = sq.sum()
    print(f"полная сумма квадратов {total:,.0f}, MSE {total / len(truth):,.0f}\n")

    groups = {
        "скрытая метка 10000 (неактивные)": truth == 10_000,
        "скрытая метка 0 (нет CC50)": truth == 0,
        "активные, значение из литературы": (truth != 10_000) & (truth != 0) & frame["matched"].eq(1).to_numpy(),
        "активные, предсказание модели": (truth != 10_000) & (truth != 0) & frame["matched"].eq(0).to_numpy(),
    }
    print(f"{'группа':38} {'строк':>6} {'доля SSE':>9} {'MSE в группе':>14}")
    for name, mask in groups.items():
        print(f"{name:38} {mask.sum():6d} {sq[mask].sum() / total:9.1%} {sq[mask].mean():14,.0f}")

    print("\nпропущенные неактивные (скрытая метка 10000, предсказано мало):")
    miss = (truth == 10_000)
    order = np.argsort(-sq * miss)[:12]
    view = pd.DataFrame({
        "Title": test["Title"].to_numpy()[order],
        "предсказано": sub[order].round(0),
        "matched": frame["matched"].to_numpy()[order],
        "lit_si": frame["literature_si"].to_numpy()[order],
        "вклад": (sq[order] / total * 100).round(1),
    })
    print(view.to_string(index=False))

    print("\nсамые дорогие строки вообще:")
    order = np.argsort(-sq)[:15]
    view = pd.DataFrame({
        "Title": test["Title"].to_numpy()[order],
        "скрытая метка": truth[order].round(1),
        "предсказано": sub[order].round(1),
        "matched": frame["matched"].to_numpy()[order],
        "вклад %": (sq[order] / total * 100).round(1),
    })
    print(view.to_string(index=False))

    lit = frame["matched"].eq(1).to_numpy() & (truth != 10_000) & (truth != 0)
    print(f"\nлитература против скрытой метки на {lit.sum()} активных строках:")
    err = np.log10(np.clip(sub[lit], 1e-6, None)) - np.log10(np.clip(truth[lit], 1e-6, None))
    print(f"  медианная |ошибка| в log10 {np.median(np.abs(err)):.3f}, в пределах 2x {np.mean(np.abs(err) <= np.log10(2)):.1%}")
    nolit = frame["matched"].eq(0).to_numpy() & (truth != 10_000) & (truth != 0)
    err2 = np.log10(np.clip(sub[nolit], 1e-6, None)) - np.log10(np.clip(truth[nolit], 1e-6, None))
    print(f"модель против скрытой метки на {nolit.sum()} активных строках без литературы:")
    print(f"  медианная |ошибка| в log10 {np.median(np.abs(err2)):.3f}, в пределах 2x {np.mean(np.abs(err2) <= np.log10(2)):.1%}")

    mism = frame["matched"].eq(1).to_numpy() & (truth == 10_000)
    print(f"\nнайдено в литературе, но скрытая метка 10000: {mism.sum()} строк, "
          f"вклад {sq[mism].sum() / total:.1%}")
    opp = frame["matched"].eq(0).to_numpy() & (truth != 10_000)
    print(f"не найдено в литературе, но скрытая метка не 10000: {opp.sum()} строк, "
          f"вклад {sq[opp].sum() / total:.1%}")


if __name__ == "__main__":
    main()
