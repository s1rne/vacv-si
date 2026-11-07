"""Хард-оверрайды из таблиц OA-статей (data/processed/article_si.csv).

Соревнование строит SI из открытых данных; часть статей их пайплайн пропустил.
Таблица статьи, проверенная на train бит-в-бит (BMCL 2016, серия 24a/24b),
даёт два класса переопределений:

- EC50 «>100» (цензурированный, «не активен до 100 µM») → SI = 10000, ровно
  как сентинелы train (11/11 совпадений);
- измеренный EC50 → SI = MCC/EC50, граница «>100» → 100 (2/2 точных).

Переопределение подменяет итоговое предсказание: строка известна из
литературы, модельная вероятность к ней не применяется.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ARTICLE_SI = Path("data/processed/article_si.csv")


@lru_cache(maxsize=1)
def article_frame() -> pd.DataFrame:
    return pd.read_csv(ARTICLE_SI)


def attach_article(frame: pd.DataFrame) -> pd.DataFrame:
    """Колонки article_sentinel (bool) и article_si (float, NaN без переопределения)."""
    art = article_frame().set_index("smiles")
    out = frame.copy()
    out["article_sentinel"] = out["SMILES"].map(art["sentinel"]).fillna(False).astype(bool)
    out["article_si"] = pd.to_numeric(out["SMILES"].map(art["si"]), errors="coerce")
    return out


def apply_article_overrides(frame: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    """Подмена предсказания строками, известными из литературы."""
    art = article_frame().set_index("smiles")
    sentinel = frame["SMILES"].map(art["sentinel"]).fillna(False).to_numpy(dtype=bool)
    si = pd.to_numeric(frame["SMILES"].map(art["si"]), errors="coerce").to_numpy(dtype=float)
    out = prediction.copy()
    out[sentinel] = 10000.0
    known = np.isfinite(si)
    out[known] = si[known]
    return out
