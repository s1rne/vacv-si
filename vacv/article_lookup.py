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
#: Table 1/2 Russ. Chem. Bull. 2019 (Nikitina, 10.1007/s11172-019-2467-6):
#: 1-гидрокси-2-(2-гидроксифенил)имидазолы, 5/5 train сентинелов бит-в-бит
#: (см. experiments/rcb_si.py).
RCB_SI = Path("data/processed/rcb_si.csv")
#: Table 1 ChemMedChem 2022 (Sokolova, 10.1002/cmdc.202100771): борнил/фенхил
#: амиды серий 'as-'/'qs-', 11/11 train-строк бит-в-бит (см. experiments/cmdc_si.py).
CMDC_SI = Path("data/processed/cmdc_si.csv")
#: Table 4 RSC Med Chem 2024 (Nikitina, 10.1039/d4md00181h): 2-арил-1-гидрокси-
#: имидазолы (ряды 4-13), метки соревнования сдвинуты на ряд — матч по InChIKey,
#: 12/12 train-строк бит-в-бит (см. experiments/rsc2024_si.py).
RSC2024_SI = Path("data/processed/rsc2024_si.csv")
#: Table 1 MedChemComm 2018 (Kovaleva, PMC6335999): камфорные N-ацилгидразоны,
#: матч по InChIKey структур ChEMBL-дока (CHEMBL4368876); train#226 '10'
#: 25.928 бит-в-бит (см. experiments/kovaleva_si.py).
KOVALEVA_SI = Path("data/processed/kovaleva_si.csv")
#: JMC 1984 (10.1021/jm00377a002) Table II: sid103201900 (3-деазагуанин),
#: CC50 = toxic_level×1000/MW = 14.65, IC50 = VR(vaccinia) = 1.55 → 9.452,
#: бит-в-бит на train#97 (см. experiments/sid3dg_rule.py).
SID3DG_SI = Path("data/processed/sid3dg_si.csv")
#: Table 2 Molecules 2022 (PMC9331314): камфорные тиосемикарбазоны 2/2′,
#: тиазолидин-4-оны 3/3′/4, дигидротиазолы 5-8; 10/10 train бит-в-бит
#: (см. experiments/molecules2022_si.py).
MOLECULES2022_SI = Path("data/processed/molecules2022_si.csv")
#: Table 1 Chem. Biodiversity 2018 (10.1002/cbdv.201800153): борнил-эфиры/амиды
#: N-гетероароматических кислот (4-40) + фенхол '28'/нол '29'; 15/15 train
#: бит-в-бит (см. experiments/cbdv2018_si.py).
CBDV2018_SI = Path("data/processed/cbdv2018_si.csv")
#: Antiviral Research 2021 (10.1016/j.antiviral.2021.105086, PMC9629033):
#: метиленовый синий PAV-866/PAV-825/PAV-661, EC50 из Fig. 2B + CC50=15.0
#: (цензура к макс. концентрации); 2/2 train бит-в-бит
#: (см. experiments/av2021_si.py).
AV2021_SI = Path("data/processed/av2021_si.csv")
#: Table 1 Front. Microbiol. 2020 (10.3389/fmicb.2020.00603, PMC7190985):
#: аналоги Retro-1 PA104/PA63/PA24/Retro1, IC50 из Table 1 + CC50=400.0
#: (цензура ">400"); 2/2 train бит-в-бит (см. experiments/retro2020_si.py).
RETRO2020_SI = Path("data/processed/retro2020_si.csv")


@lru_cache(maxsize=1)
def article_frame() -> pd.DataFrame:
    frames = [pd.read_csv(ARTICLE_SI)]
    for extra in (PAPER_SI, LIT_SI, BMCL_SI, RCB_SI, CMDC_SI, RSC2024_SI,
                  KOVALEVA_SI, SID3DG_SI, MOLECULES2022_SI, CBDV2018_SI,
                  AV2021_SI, RETRO2020_SI):
        if extra.exists():
            frames.append(pd.read_csv(extra))
    out = pd.concat(frames, ignore_index=True)
    # дубли (smiles, Title) — первый файл имеет приоритет (24a/24b выверены
    # точнее); дубли smiles с РАЗНЫМИ Title легитимны: один SMILES может быть
    # покрыт из разных строк соревнования с разными источниками
    # (test#172 '11' PMC9533879=97.78 vs test#332 '15a' PMC9865139=3192.3)
    out = out.drop_duplicates(["smiles", "Title"], keep="first")
    return out


def override_lookup(art: pd.DataFrame):
    """lookup(smiles, Title) → (sentinel, si) | None.

    Приоритет — точное совпадение (smiles, Title); иначе первое совпадение по
    smiles, НО только если Title не встречается в таблице с другой структурой
    (иначе fallback подменяет легитимные строки-коллизии: train#217 '7' —
    тот же SMILES, что train#33 '36' cbdv-2018, но другая статья с y=10000).
    Fallback нужен для test-дублей SMILES из разных источников.
    """
    art = art.copy()
    if "Title" in art.columns:
        art["Title"] = art["Title"].astype(str)
    else:
        art["Title"] = ""
    art = art.drop_duplicates(["smiles", "Title"], keep="first")
    si = pd.to_numeric(art["si"], errors="coerce")
    pair = {(s, t): (bool(sn), v) for s, t, sn, v in
            zip(art["smiles"], art["Title"], art["sentinel"], si)}
    known_titles = {t.strip() for t in art["Title"]}
    first: dict[str, tuple[bool, float]] = {}
    for s, sn, v in zip(art["smiles"], art["sentinel"], si):
        first.setdefault(s, (bool(sn), v))

    def lookup(smiles: str, title: str) -> tuple[bool, float] | None:
        t = str(title)
        hit = pair.get((smiles, t))
        if hit is not None:
            return hit
        if t.strip() in known_titles:
            return None  # Title принадлежит другой структуре — не подменяем
        return first.get(smiles)

    return lookup


def _attach(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lookup = override_lookup(article_frame())
    titles = frame["Title"].astype(str) if "Title" in frame.columns else \
        pd.Series([""] * len(frame), index=frame.index)
    sent, si = [], []
    for s, t in zip(frame["SMILES"], titles):
        hit = lookup(s, t)
        sent.append(hit[0] if hit else False)
        si.append(hit[1] if hit else np.nan)
    return np.asarray(sent, dtype=bool), np.asarray(si, dtype=float)


def attach_article(frame: pd.DataFrame) -> pd.DataFrame:
    """Колонки article_sentinel (bool) и article_si (float, NaN без переопределения)."""
    out = frame.copy()
    sent, si = _attach(out)
    out["article_sentinel"] = sent
    out["article_si"] = si
    return out


def apply_article_overrides(frame: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    """Подмена предсказания строками, известными из литературы."""
    sentinel, si = _attach(frame)
    out = prediction.copy()
    out[sentinel] = 10000.0
    known = np.isfinite(si)
    out[known] = si[known]
    return out
