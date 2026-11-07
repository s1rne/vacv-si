"""Хард-оверрайды из таблиц OA-статей (article_si.csv + paper_si.csv).

Соревнование строит SI из открытых данных; часть статей их пайплайн пропустил.
article_si.csv — BMCL 2016 (серия 24a/24b), выверена на train бит-в-бит:
EC50 «>100» → сентинел 10000 (11/11), измеренный EC50 → SI = MCC/EC50 (2/2).
paper_si.csv — ещё пять статей (RSC Med Chem 2024, MedChemComm 2018,
RSC Med Chem 2020, EJMC 2021, Viruses 2022), сматченные через ChEMBL-значения
и лейбл-схему, см. experiments/paper_si_match.py.

Переопределение подменяет итоговое предсказание: строка известна из
литературы, модельная вероятность к ней не применяется.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ARTICLE_SI = Path("data/processed/article_si.csv")
#: Таблицы ещё пяти OA-статей (см. experiments/paper_si_match.py): матчинг
#: через ChEMBL-значения с валидацией на train + лейбл-схема Viruses 2022,
#: подтверждённая бит-в-бит на train (13a/13b/17b).
PAPER_SI = Path("data/processed/paper_si.csv")
#: Прямые SI-записи ChEMBL (Ratio CC50/IC50 → пара) и сентинелы калиброванных
#: доков (см. experiments/lit_si_rule.py; 17/17 бит-в-бит на train).
LIT_SI = Path("data/processed/lit_si.csv")
#: Table 1 BMCL 2023 (Nikitina, PMC9674569): 1-гидроксиимидазолы 1a-f/2a-l,
#: 5/5 train-строк бит-в-бит (см. experiments/bmcl_si.py).
BMCL_SI = Path("data/processed/bmcl_si.csv")


@lru_cache(maxsize=1)
def article_frame() -> pd.DataFrame:
    frames = [pd.read_csv(ARTICLE_SI)]
    for extra in (PAPER_SI, LIT_SI, BMCL_SI):
        if extra.exists():
            frames.append(pd.read_csv(extra))
    out = pd.concat(frames, ignore_index=True)
    if out["smiles"].duplicated().any():
        # первый файл имеет приоритет (24a/24b выверены точнее)
        out = out.drop_duplicates("smiles", keep="first")
    return out


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
