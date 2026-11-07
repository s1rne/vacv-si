"""Оверрайды из Table 1 BMCL 2023 (Nikitina et al., 10.1016/j.bmcl.2022.129080,
PMC9674569): цитотоксичность/активность 1-гидроксиимидазолов 1a-f, 2a-l против
VACV (Copenhagen) в Vero.

Структуры собираются из описания серии (R1 = арил у C2, R2 = OEt / NHBu /
циклопропиламид), матч к строкам соревнования по InChIKey. Валидация: 5 train-строк
бит-в-бит (1c, 1f y=919.914, 2c y=5.946, 2g y=9.324, 2h сентинел).

test#486 '1a' (структура 1b) НЕ вливается: в train дубликаты структур имеют
разные y ('36' 39.984 vs '7' 10000; idoxuridine 0.152 vs 11.905), т.е. y
ключуется по строке источника, а метка '1a' может принадлежать другому документу
(напр., ChemMedChem 2022), где 1a измерена иначе.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

RDLogger.DisableLog("rdApp.*")

OUT = Path("data/processed/bmcl_si.csv")

# (label, R1, R2, CC50 µM raw, IC50 µM raw) — Table 1, VACV Copenhagen / Vero
TABLE1 = [
    ("1a", "2-OH",   None,          "1825.57", "15.53"),
    ("1b", "H",      None,          "171.54",  "N/A"),
    ("1c", "2-NO2",  None,          "343.30",  "N/A"),
    ("1d", "3-NO2",  None,          "374.91",  "N/A"),
    ("1e", "4-NO2",  None,          "45.70",   "0.45"),
    ("1f", "4-CF3",  None,          "321.97",  "0.35"),
    ("2a", "2-OH",   "C4H9",        "196.89",  "15.02"),
    ("2b", "2-OH",   "cyclo-C3H5",  "298.90",  "N/A"),
    ("2c", "H",      "C4H9",        "436.26",  "73.37"),
    ("2d", "H",      "cyclo-C3H5",  ">389.11", "N/A"),
    ("2e", "2-NO2",  "C4H9",        ">314.47", "N/A"),
    ("2f", "2-NO2",  "cyclo-C3H5",  ">331.13", "N/A"),
    ("2g", "3-NO2",  "C4H9",        "164.47",  "17.64"),
    ("2h", "3-NO2",  "cyclo-C3H5",  "260.93",  "N/A"),
    ("2i", "4-NO2",  "C4H9",        "200.63",  "3.02"),
    ("2j", "4-NO2",  "cyclo-C3H5",  "356.95",  "10.73"),
    ("2k", "4-CF3",  "C4H9",        "26.10",   "3.40"),
    ("2l", "4-CF3",  "cyclo-C3H5",  "201.85",  "20.58"),
]
ARYL = {"H": "c2ccccc2", "2-OH": "c2ccccc2O", "2-NO2": "c2ccccc2[N+](=O)[O-]",
        "3-NO2": "c2cc(ccc2)[N+](=O)[O-]", "4-NO2": "c2ccc([N+](=O)[O-])cc2",
        "4-CF3": "c2ccc(C(F)(F)F)cc2"}
TAIL = {None: "C(=O)OCC", "C4H9": "C(=O)NCCCC", "cyclo-C3H5": "C(=O)NC3CC3"}
#: строки, матч которых неоднозначен (см. docstring): держим строку соревнования
#: с голой меткой '1a' (test#486, структура 1b); 'PANI-17' — та же структура 1a,
#: вливается
HOLD_TITLES = {"1a"}

PMC = "PMC9674569"


def build_smiles(r1: str, r2) -> str:
    tail = TAIL[r2]
    return f"c1({ARYL[r1]})nc(C)c({tail})n1O"


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
    for lab, r1, r2, cc_raw, ic_raw in TABLE1:
        mol = Chem.MolFromSmiles(build_smiles(r1, r2))
        key = InchiToInchiKey(MolToInchi(mol))
        hits = key_to_rows.get(key, [])
        if not hits:
            continue
        sentinel = ic_raw == "N/A"
        si = None if sentinel else float(cc_raw.lstrip(">")) / float(ic_raw)
        for split, i, title, smiles in hits:
            if title.strip() in HOLD_TITLES:
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
                         "pmc": PMC, "label": lab, "rule": "bmcl_table1"})

    print(f"train-проверок бит-в-бит: {n_ok}/{n_total}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    n_sent = int(out["sentinel"].sum())
    print(f"→ {OUT}: {len(out)} строк (сентинелов {n_sent}, активных {len(out) - n_sent})")


if __name__ == "__main__":
    main()
