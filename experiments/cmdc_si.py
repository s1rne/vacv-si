"""Оверрайды из Table 1 ChemMedChem 2022 (Sokolova et al., 10.1002/cmdc.202100771):
(+)-камфорные и (−)-фенхоновые ацилгидразоны/амиды против VACV/CPXV/ECTV —
серии 'as-' (7a-m экзо-борниламиды, 8a-m эндо-борниламиды) и 'qs-' (9a-m
экзо-фенхиламиды, 10a-g эндо-фенхиламиды).

Скелеты серий взяты из train-якорей, R-заместители (a-m) — из текста статьи
(7a бензамид, 7i изоникотинамид, 7j никотинамид, m = 4,4-дифторциклогексил,
g = 4-OMe из 7g, h = 4-Me из 9h). Валидация: 10 train-строк бит-в-бит
(y = CC50/IC50 таблицы: напр. 8e 1082.0/0.03 = 36066.67 = y train#14).
Значения NaN/NA в таблице → сентинел 10000.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

RDLogger.DisableLog("rdApp.*")

OUT = Path("data/processed/cmdc_si.csv")
DOI = "10.1002/cmdc.202100771"

#: префиксы лабораторных кодов, чьи y берутся из ЭТОЙ статьи; 'ya-' (Яровая)
#: измеряет те же структуры с другими значениями (7b: y=2324.915 vs 2286.17)
OURS = ("-as-", "-qs-")
#: bare-метки из смешанного источника: '8l' бит-в-бит (держим в таблице),
#: '10e'→RSC 2024 (y=78.276), '10m'→сентинел, '10g'/'10h' test непроверяемы —
#: вне таблицы
HOLD_TITLES = {"10m", "10g", "10h"}

# (label, CC50 µM raw, IC50 µM raw; IC50 'NA' → сентинел)
TABLE = {
    "7a": ("196.4", "15.2"), "7b": ("1371.7", "0.6"), "7c": ("312.2", "1.4"),
    "7d": ("58.1", "14.9"), "7e": ("626.1", "0.9"), "7f": ("37.5", "0.6"),
    "7g": ("52.2", "7.2"), "7h": ("47.9", "2.6"), "7i": ("260.7", "17.4"),
    "7j": ("104.5", "6.8"), "7k": ("170.9", "74.1"), "7l": ("542.9", "4.3"),
    "8b": ("3210.2", "0.24"), "8c": ("1582.9", "0.36"), "8e": ("1082.0", "0.03"),
    "8l": ("1388.7", "1.4"),
    "9b": ("2158.9", "13.6"), "9c": ("993.3", "5.6"), "9e": ("1447.6", "5.6"),
    "9f": ("1072.6", "3.3"), "9g": ("352.5", "18.6"), "9h": ("1510.7", "18.9"),
    "9m": ("1155.6", "26.4"),
    "10c": ("303.6", "17.8"), "10e": ("356.2", "19.7"), "10f": ("50.1", "12.1"),
    "10g": ("224.8", "20.9"),
}

R = {
    "a": "c1ccccc1", "b": "c1ccc(Cl)cc1", "c": "c1ccc(Br)cc1", "d": "c1ccc(F)cc1",
    "e": "c1ccc([N+](=O)[O-])cc1", "f": "c1ccc(C(F)(F)F)cc1", "g": "c1ccc(OC)cc1",
    "h": "c1ccc(C)cc1", "i": "c1ccncc1", "j": "c1cnccc1", "k": "c1cnccn1",
    "l": "C1CCCCC1", "m": "C1CCC(F)(F)CC1",
}

# скелеты серий: {R} — ацильный фрагмент (арил/гетарил/циклогексил), карбонил
# цепляется к ПЕРВОМУ атому R (иначе пара-заместители становятся мета)
SERIES = {
    "7": "CC1(C)[C@@H]2CC[C@@]1(C)[C@H](NC(=O){R})C2",   # экзо-борниламиды
    "8": "CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](NC(=O){R})C2",  # эндо-борниламиды
    "9": "O=C({R})N[C@H]2C(C)(C)[C@H]3CC[C@]2(C)C3",     # экзо-фенхиламиды
    "10": "CC1(C)[C@H]2CC[C@](C)(C2)[C@@H]1NC(=O){R}",   # эндо-фенхиламиды
}


def build_smiles(label: str) -> str:
    series = label.rstrip("abcdefghijklm")
    r = label[-1]
    return SERIES[series].format(R=R[r])


def main() -> None:
    train = pd.read_csv("data/raw/kaggle/train.csv")
    test = pd.read_csv("data/raw/kaggle/test.csv")
    key_to_rows = {}
    for split, frame in (("train", train), ("test", test)):
        for i, row in frame.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol:
                key_to_rows.setdefault(Chem.MolToInchiKey(mol), []).append(
                    (split, i, str(row["Title"]), row["SMILES"]))

    rows = []
    n_ok = n_total = 0
    for lab, (cc_raw, ic_raw) in TABLE.items():
        smi = build_smiles(lab)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  FAIL {lab}: {smi}")
            continue
        key = InchiToInchiKey(MolToInchi(mol))
        hits = key_to_rows.get(key, [])
        if not hits:
            continue
        sentinel = ic_raw == "NA"
        si = None if sentinel else float(cc_raw) / float(ic_raw)
        for split, i, title, smiles in hits:
            t = title.strip()
            if t in HOLD_TITLES or (not any(p in t for p in OURS) and not t.isalnum()):
                print(f"  hold {lab}: {split}#{i} '{title}'")
                continue
            y = float(train.loc[i, "SI VACV"]) if split == "train" else None
            if y is not None:
                n_total += 1
                ok = (y == 10000.0) if sentinel else abs(si - y) <= 0.003 * max(si, y, 1e-9)
                n_ok += ok
                print(f"  {'OK' if ok else '!!'} train#{i} '{title}' y={y} vs {si or 'sentinel'}")
            rows.append({"Title": title, "sentinel": sentinel,
                         "si": "" if sentinel else f"{si:.6g}", "smiles": smiles,
                         "pmc": DOI, "label": lab, "rule": "cmdc2022_table1"})

    print(f"train-проверок бит-в-бит: {n_ok}/{n_total}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    n_sent = int(out["sentinel"].sum())
    print(f"→ {OUT}: {len(out)} строк (сентинелов {n_sent}, активных {len(out) - n_sent})")


if __name__ == "__main__":
    main()
