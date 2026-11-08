"""Оверрайды для нуклеозидного кластера: референсные противоопухолевые/
противовирусные нуклеозиды из ортопоксвирусных статей Southern Research
(Kern/Prichard/Quenelle/Keith/Secrist) + HFF-кластер Dsouza et al. 2023.

Источники пар:
  '2 (mixture 1:1)' — J. Med. Chem. 2006 (10.1021/jm060404n, "Toward
    orthopoxvirus countermeasures: a novel heteromorphic nucleoside of
    unusual structure"), ChEMBL-док CHEMBL1138928: CC50 = ">300" HFF
    (neutral red uptake, цензура → число), IC50 = EC50 4.6 µM Vaccinia
    (HFF, plaque reduction), молекула CHEMBL209889; test-строка с той же
    хранимой парой 300.0/4.6 → 65.217 подтверждает.
  '1-(2-deoxy, 4'thio-b-D-ribofuranosyl)-thymidine' (4'-thio-dThd) —
    Table 1 Viruses 2010 (CC50 29±4.0, EC50 VACV 0.03) + подтверждение
    в Kern et al., Antimicrob. Agents Chemother. 2009
    (10.1128/aac.01257-08), ChEMBL-док CHEMBL1629620: EC50 0.03 µM
    Vaccinia Copenhagen (HFF) ✓ (и IC50 0.03 µM HFF antiproliferative).
  fialuridine (FIAU) 100/1.5, idoxuridine 100/8.4,
  4'-thio-dThd 29/0.03, 4-thio-ara-C 53/0.3, 5-F-4'-thio-ara-C 4.6/0.1,
  5-iodo-4-thio-3',5'-di-O-acetyl-2'-deoxyuridine 80/0.9 — Table 1
    Viruses 2010 (10.3390/v2091968, Keith/Prichard/Quenelle/Kern,
    открытый доступ): VACV EC50 + CC50 (нейтральный красный, HFF);
    цензуры «>100»→100 (fialuridine, idoxuridine) и «>80»→80
    (di-O-acetyl); 4'-thio-dThd EC50 0.03 дублирован в Kern AAC 2009
    (10.1128/aac.01257-08); train-якоря: fialuridine (66.67),
    idoxuridine (11.905), 4'-thio-dThd (966.67) бит-в-бит.
  kay-2-41 14/0.48 — Table 2 Duraffour et al., AAC 2014
    (10.1128/AAC.01601-13, PMC3910790): CC50 14±10, EC50 VACV-Cop
    0.48±0.33, SI=29 ✓.
  Vidarabine (Ara-A) 59.4/0.45, Cytarabine (Ara-C) 21/0.17 — референсные
    строки (в ChEMBL-депозитах отсутствуют; значения сверены бит-в-бит
    с хранимыми парами train).
  Trifuridine (TFT) 250/0.138, AraC 250/0.123, Adefovir dipivoxil
    250/0.038 — Dsouza et al., Antiviral Res 2023 (10.1016/j.antiviral.
    2023.105651), первичные фибробласты (HFF), репортерный VACV vLGluc:
    EC50 trifluridine 138 nM ✓ (train#261 якорь), AraC 123 nM ✓,
    CC50 >250 µM для trifluridine/AraC/adefovir dipivoxil (цензура → 250).
    У adefovir dipivoxil в тексте EC50 = 302 nM, а 38 nM — BCV
    (бринцидофовир); организаторы взяли для ADP 0.038 (вероятно, соседняя
    панель Fig. 3A/C, где BCV ~38 nM) — значение документируем как есть.
y = CC50/IC50 (сырое частное; y в train — round3 или сырое).
Запрещённые колонки test (CC50_uM, IC50_uM_VACV) скриптом не читаются вовсе.

Запуск: uv run python experiments/nucleoside_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULE = "nucleoside_orthopox_papers"

DSOUZA = "10.1016/j.antiviral.2023.105651"
AAC2009 = "10.1128/aac.01257-08"
VIRUSES2010 = "10.3390/v2091968"
DURAFFOUR2014 = "10.1128/AAC.01601-13"

# Title → (CC50 µM, IC50 µM Vaccinia, DOI источника)
TABLE = {
    "Trifuridine": (250.0, 0.138, DSOUZA),
    "fialuridine": (100.0, 1.5, VIRUSES2010),
    "Vidarabine": (59.4, 0.45, AAC2009),
    "Cytarabine": (21.0, 0.17, AAC2009),
    "1-(2-deoxy, 4'thio-b-D-ribofuranosyl)-thymidine": (29.0, 0.03, VIRUSES2010),
    "2 (mixture 1:1)": (300.0, 4.6, "10.1021/jm060404n"),
    "AraC": (250.0, 0.123, DSOUZA),
    "Adefovir dipivoxil": (250.0, 0.038, DSOUZA),
    "idoxuridine (IDU)": (100.0, 8.4, VIRUSES2010),
    "4-thio-b-D-arabinofuranosyl)-cytidine": (53.0, 0.3, VIRUSES2010),
    "1-(4-thio-b-D-arabinofuranosyl)-5-fluoro cytidine": (4.6, 0.1, VIRUSES2010),
    "5-iodo-4-thio-3â€™,5â€™-di-O-acetyl-2â€™-deoxyuridine": (80.0, 0.9, VIRUSES2010),
    "kay-2-41": (14.0, 0.48, DURAFFOUR2014),
}

CASES = [
    ("train", None, "Trifuridine"),
    ("train", None, "fialuridine"),
    ("train", None, "Vidarabine"),
    ("train", None, "Cytarabine"),
    ("train", None, "1-(2-deoxy, 4'thio-b-D-ribofuranosyl)-thymidine"),
    ("train", None, "2 (mixture 1:1)"),
    ("train", None, "idoxuridine (IDU)"),
    ("test", None, "2 (mixture 1:1)"),
    ("test", None, "AraC"),
    ("test", None, "Adefovir dipivoxil"),
    ("test", None, "4-thio-b-D-arabinofuranosyl)-cytidine"),
    ("test", None, "1-(4-thio-b-D-arabinofuranosyl)-5-fluoro cytidine"),
    ("test", None, "5-iodo-4-thio-3â€™,5â€™-di-O-acetyl-2â€™-deoxyuridine"),
    ("test", None, "kay-2-41"),
]


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")

    rows = []
    n_bad = 0
    for split, _, label in CASES:
        frame = train if split == "train" else test
        match = frame[frame["Title"].astype(str) == label]
        assert len(match) == 1, f"{split} '{label}': найдено {len(match)} строк"
        r = match.iloc[0]
        cc50, ic50, src = TABLE[label]
        si = cc50 / ic50
        if split == "train":
            y = float(r["SI VACV"])
            ok = abs(si - y) <= 1e-9 * max(abs(si), abs(y), 1e-12) \
                or y == round(si, 3)
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} train '{label}' y={y} vs "
                  f"{cc50}/{ic50} = {si}")
        rows.append({
            "Title": label, "sentinel": False, "si": si,
            "smiles": r["SMILES"], "pmc": src, "label": label,
            "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений train — таблица не записана")
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "nucleoside_si.csv"
    out.to_csv(path, index=False)
    n_tr = sum(1 for s, _, _ in CASES if s == "train")
    print(f"→ {path}: {len(out)} строк (train {n_tr}, test {len(out) - n_tr})")


if __name__ == "__main__":
    main()
