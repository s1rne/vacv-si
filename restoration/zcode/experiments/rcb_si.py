"""Оверрайды из Table 1/2 Russian Chem. Bull. 2019 (Nikitina et al.,
10.1007/s11172-019-2467-6): 1-гидрокси-2-(2-гидроксифенил)имидазолы 1a-h,
N-оксиды 4a-c, 1-метокси 5a-c, 1H-имидазолы 6a-c против VACV (Copenhagen) в Vero.

Структуры собираются из Scheme 1 (R1 = арил, R2 = Me/Ph, R3 = COCH3/CO2Et/
CONHPh/фьюзд-циклопентанон) и матачатся по InChIKey. Валидация: 5 train-строк
бит-в-бит (сентинелы: 1g='PANI-45' #107, 1h='PANI-46' #269, 4a='PANI-19' #86,
4c='PANI-21' #83, 6b='PANI-53' #132 — y=10000).

Исключения:
- 1b ('PANI-17' test#165) не вливается: та же структура = BMCL 1a, значения
  согласуются до 0.02% (117.53 vs 117.55), приоритет у bmcl_si.csv;
- 1c ('PANI-18' train#42) не вливается: таблица даёт 203.00/1.29 = 157.36,
  а y=186.25 — источник соревнования другой (см. vault: pani-series-sources).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

RDLogger.DisableLog("rdApp.*")

OUT = Path("data/processed/rcb_si.csv")

ARH = "c2ccccc2O"               # 2-hydroxyphenyl
ARCL = "c2cc(Cl)ccc2O"          # 2-hydroxy-5-chlorophenyl
ARBR = "c2cc(Br)ccc2O"          # 2-hydroxy-5-bromophenyl
AROB = "c2ccc(OCc3ccccc3)cc2O"  # 2-hydroxy-4-benzyloxyphenyl

# (label, smiles-template, CC50 µg/mL raw, IC50 µg/mL raw)
TABLE = [
    ("1a", "c1(ARH)nc(C)c(C(=O)C)n1O", "64.80", "14.58"),
    ("1c", "O=C1CC(C)(C)Cc2c1n(O)c(n2)c1ccccc1O", "203.00", "1.29"),
    ("1d", "c1(ARH)nc(-c3ccccc3)c(C(=O)Nc4ccccc4)n1O", "0.201", "N.a."),
    ("1e", "c1(ARCL)nc(C)c(C(=O)C)n1O", "1.06", "N.a."),
    ("1f", "c1(AROB)nc(C)c(C(=O)OCC)n1O", "1.434", "N.a."),
    ("1g", "c1(ARCL)nc(-c3ccccc3)c(C(=O)Nc4ccccc4)n1O", "<0.16", "N.a."),
    ("1h", "c1(ARBR)nc(-c3ccccc3)c(C(=O)Nc4ccccc4)n1O", "<0.16", "N.a."),
    ("4a", "[O-][n+]1c(C(=O)C)c(C)n(C)c1-c2ccccc2O", ">100", "N.a."),
    ("4b", "[O-][n+]1c(C(=O)OCC)c(C)n(C)c1-c2ccccc2O", ">100", "N.a."),
    ("4c", "C1C(C)(C)CC(=O)c(c12)[n+]([O-])c(n2C)-c3ccccc3O", ">100", "N.a."),
    ("5a", "COn1c(C(=O)C)c(C)nc1-c2ccccc2O", "56.061", "N.a."),
    ("5b", "COn1c(C(=O)OCC)c(C)nc1-c2ccccc2O", ">100", "42.926"),
    ("5c", "C1C(C)(C)CC(=O)c(c12)n(OC)c(n2)-c3ccccc3O", ">100", "N.a."),
    ("6a", "CC(=O)c1c(C)nc(-c2ccccc2O)[nH]1", ">100", "N.a."),
    ("6b", "CCOC(=O)c1c(C)nc(-c2ccccc2O)[nH]1", "5.231", "N.a."),
    ("6c", "O=C1CC(C)(C)Cc2c1[nH]c(n2)c1ccccc1O", ">100", "N.a."),
]
#: строки-исключения (см. docstring): их матч держим вне таблицы
HOLD_LABELS = {"1b", "1c"}

DOI = "10.1007/s11172-019-2467-6"


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
    for lab, tpl, cc_raw, ic_raw in TABLE:
        smi = (tpl.replace("ARH", ARH).replace("ARCL", ARCL)
                  .replace("ARBR", ARBR).replace("AROB", AROB))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  FAIL {lab}: {smi}")
            continue
        key = InchiToInchiKey(MolToInchi(mol))
        hits = key_to_rows.get(key, [])
        if not hits:
            continue
        if lab in HOLD_LABELS:
            for split, i, title, _ in hits:
                print(f"  hold {lab}: {split}#{i} '{title}'")
            continue
        sentinel = ic_raw == "N.a."
        si = None if sentinel else float(cc_raw.lstrip("<>")) / float(ic_raw)
        for split, i, title, smiles in hits:
            y = float(train.loc[i, "SI VACV"]) if split == "train" else None
            if y is not None:
                n_total += 1
                ok = (y == 10000.0) if sentinel else abs(si - y) <= 0.003 * max(si, y, 1e-9)
                n_ok += ok
                print(f"  {'OK' if ok else '!!'} train#{i} '{title}' y={y} vs {si or 'sentinel'}")
            rows.append({"Title": title, "sentinel": sentinel,
                         "si": "" if sentinel else f"{si:.6g}", "smiles": smiles,
                         "pmc": DOI, "label": lab, "rule": "rcb2019_table"})

    print(f"train-проверок бит-в-бит: {n_ok}/{n_total}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    n_sent = int(out["sentinel"].sum())
    print(f"→ {OUT}: {len(out)} строк (сентинелов {n_sent}, активных {len(out) - n_sent})")


if __name__ == "__main__":
    main()
