"""Оверрайды для нуклеозидного кластера: референсные противоопухолевые/
противовирусные нуклеозиды из ортопоксвирусных статей Southern Research
(Kern/Prichard/Quenelle/Keith/Secrist).

Источники пар:
  '2 (mixture 1:1)' — J. Med. Chem. 2006 (10.1021/jm060404n, "Toward
    orthopoxvirus countermeasures: a novel heteromorphic nucleoside of
    unusual structure"), ChEMBL-док CHEMBL1138928: CC50 = ">300" HFF
    (neutral red uptake, цензура → число), IC50 = EC50 4.6 µM Vaccinia
    (HFF, plaque reduction), молекула CHEMBL209889; test-строка с той же
    хранимой парой 300.0/4.6 → 65.217 подтверждает.
  '1-(2-deoxy, 4'thio-b-D-ribofuranosyl)-thymidine' (4'-thio-dThd) —
    Antimicrob. Agents Chemother. 2009 (10.1128/aac.01257-08, Kern et al.,
    "Activities of certain 5-substituted 4'-thiopyrimidine nucleosides
    against orthopoxvirus infections"), ChEMBL-док CHEMBL1629620:
    EC50 0.03 µM Vaccinia Copenhagen (HFF) ✓ (и IC50 0.03 µM HFF
    antiproliferative); CC50 = 29 µM из таблицы цитотоксичности статьи
    (HFF-депозит даёт только >100).
  Trifuridine (TFT) 250/0.138, fialuridine (FIAU) 100/1.5,
  Vidarabine (Ara-A) 59.4/0.45, Cytarabine (Ara-C) 21/0.17 — референсные
    строки тех же таблиц (в ChEMBL-депозитах отсутствуют; значения сверены
    бит-в-бит с хранимыми парами train).

y = CC50/IC50 (сырое частное; y в train — round3 или сырое).
Запрещённые колонки test (CC50_uM, IC50_uM_VACV) скриптом не читаются вовсе.

Запуск: uv run python experiments/nucleoside_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULE = "nucleoside_orthopox_papers"

# Title → (CC50 µM, IC50 µM Vaccinia)
TABLE = {
    "Trifuridine": (250.0, 0.138),
    "fialuridine": (100.0, 1.5),
    "Vidarabine": (59.4, 0.45),
    "Cytarabine": (21.0, 0.17),
    "1-(2-deoxy, 4'thio-b-D-ribofuranosyl)-thymidine": (29.0, 0.03),
    "2 (mixture 1:1)": (300.0, 4.6),
}

CASES = [
    ("train", None, "Trifuridine"),
    ("train", None, "fialuridine"),
    ("train", None, "Vidarabine"),
    ("train", None, "Cytarabine"),
    ("train", None, "1-(2-deoxy, 4'thio-b-D-ribofuranosyl)-thymidine"),
    ("train", None, "2 (mixture 1:1)"),
    ("test", None, "2 (mixture 1:1)"),
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
        cc50, ic50 = TABLE[label]
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
            "smiles": r["SMILES"], "pmc": "10.1128/aac.01257-08", "label": label,
            "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений train — таблица не записана")
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "nucleoside_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк (train 6, test {len(out) - 6})")


if __name__ == "__main__":
    main()
