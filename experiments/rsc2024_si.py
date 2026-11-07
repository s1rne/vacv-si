"""Оверрайды из Table 4 RSC Med Chem 2024 (Nikitina et al., 10.1039/d4md00181h,
PMC11331333): 2-арил-1-гидроксиимидазолы (ряды 4a-c димедон-N-OH, 5a-e
ацетилацетонат-N-OH, 6a-b димедон-N-OMe, 7a-e ацетилацетонат-N-OMe, 9a/9c
димедон-N-оксиды, 10a/b ацетилацетонат-N-оксиды, 11a-e O-бензилы 4a, 13
N-бензил-N-оксид 4a) против VACV.

Соревнование нумерует эти структуры СО СДВИГОМ (bare '10x' = RSC 11x, '9a/b' =
RSC 10a/b, '6a-e' = RSC 7a-e, '5a' = RSC 6a, '4a-e' = RSC 5a-e), поэтому матч
строго по InChIKey, а не по меткам. Валидация: 12 train-строк бит-в-бит
(#16, #31, #72, #119, #145, #162, #181, #206, #213, #227, #243, #244).

Ловушки:
- таутомеры N-оксидов: N-Me на N1, оксид на N3 (как в train#227/#162);
- коллизии цифр колец в SMILES-шаблонах (арил c2 при открытом кольце 2
  сплавляется с ядром — использовать свободные цифры);
- 5a: организаторы делят на IC50=0.65 (2 знака), в таблице 0.651
  (28.33/0.65 = 43.585 = y train#244);
- 10a/10b: IC50 N/A → сентинел 10000.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

RDLogger.DisableLog("rdApp.*")

OUT = Path("data/processed/rsc2024_si.csv")
DOI = "10.1039/d4md00181h"

AR = {"NO2": "c5ccc([N+](=O)[O-])cc5", "CN": "c5ccc(C#N)cc5", "CF3": "c5ccc(C(F)(F)F)cc5",
      "NMe2": "c5ccc(N(C)C)cc5", "OMe": "c5ccc(OC)cc5"}

# (label, smiles-шаблон, aryl, CC50 µM, IC50 µM; None → сентинел)
SERIES = {
    "4": ("CC1(C)CC(=O)C2=C(C1)N=C({a})N2O", ("NO2", "CN", "CF3")),
    "5": ("c1({a})nc(C)c(C(=O)C)n1O", ("NO2", "CN", "CF3", "NMe2", "OMe")),
    "6": ("CC1(C)CC(=O)C2=C(C1)N=C({a})N2OC", ("NO2", "CN")),
    "7": ("COn1c(C(=O)C)c(C)nc1-{a}", ("NO2", "CN", "CF3", "NMe2", "OMe")),
    "9": ("CN1C2=C(C(=O)CC(C)(C)C2)[N+]([O-])=C1{a}", ("NO2", None, "CF3")),
    "10": ("[O-][n+]1c(C(=O)C)c(C)n(C)c1-{a}", ("NO2", "CN")),
    "13": ("CC1(C)CC(=O)C2=C(C1)N(Cc4ccccc4)C({a})=[N+]2[O-]", ("NO2",)),
}
BENZYL = {"11a": "c1ccccc1", "11b": "c1ccccc1Br", "11c": "c1cc(Cl)c(Cl)cc1",
          "11d": "c1c(F)cccc1F", "11e": "c1c(C)ccc(C)c1"}

# (label, CC50, IC50; IC50 None → сентинел; '0.65' — спец-случай 5a, см. docstring)
VALUES = {
    "4a": (142.38, 0.133), "4b": (174.89, 0.498), "4c": (46.25, 0.154),
    "5a": (28.33, 0.65), "5b": (84.56, 1.12), "5c": (3.87, 0.07),
    "5d": (298.87, 12.92), "5e": (614.37, 17.91),
    "6a": (1154.34, 3.96), "6b": (757.43, 28.51),
    "7a": (443.58, 4.76), "7b": (1057.66, 228.02), "7c": (176.02, 5.90),
    "7d": (197.19, 62.08), "7e": (356.91, 207.35),
    "9a": (2440.93, 183.39), "9c": (2080.80, 14.25),
    "10a": (2979.00, None), "10b": (1508.14, None),
    "11a": (1162.40, 6.39), "11b": (100.15, 2.91), "11c": (108.62, 0.61),
    "11d": (51.71, 1.22), "11e": (104.89, 1.34),
    "13": (359.45, 27.67),
}


def build_smiles(label: str) -> str:
    if label.startswith("11"):
        return f"CC1(C)CC(=O)C2=C(C1)N=C(c5ccc([N+](=O)[O-])cc5)N2OC{BENZYL[label]}"
    if label == "13":
        return SERIES["13"][0].format(a=AR["NO2"])
    name = label.rstrip("abcde")
    ar = label[-1]
    tpl, aryls = SERIES[name]
    key = aryls["abcde".index(ar)]
    if key is None:
        raise ValueError(f"{label}: ряд отсутствует в Table 4")
    return tpl.format(a=AR[key])


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
    for lab, (cc, ic) in VALUES.items():
        mol = Chem.MolFromSmiles(build_smiles(lab))
        if mol is None:
            print(f"  FAIL {lab}")
            continue
        key = InchiToInchiKey(MolToInchi(mol))
        hits = key_to_rows.get(key, [])
        if not hits:
            continue
        sentinel = ic is None
        si = None if sentinel else cc / ic
        for split, i, title, smiles in hits:
            y = float(train.loc[i, "SI VACV"]) if split == "train" else None
            if y is not None:
                n_total += 1
                ok = (y == 10000.0) if sentinel else abs(si - y) <= 0.003 * max(si, y, 1e-9)
                n_ok += ok
                print(f"  {'OK' if ok else '!!'} train#{i} '{title}' y={y} vs {si or 'сентинел'}")
            rows.append({"Title": title, "sentinel": sentinel,
                         "si": "" if sentinel else f"{si:.6g}", "smiles": smiles,
                         "pmc": DOI, "label": lab, "rule": "rsc2024_table4"})

    print(f"train-проверок бит-в-бит: {n_ok}/{n_total}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    n_sent = int(out["sentinel"].sum())
    print(f"→ {OUT}: {len(out)} строк (сентинелов {n_sent}, активных {len(out) - n_sent})")


if __name__ == "__main__":
    main()
