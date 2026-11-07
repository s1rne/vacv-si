"""Хард-оверрайды для серий N-арилмочевин и ацилгидразидов камфоримида.

Серия 3a-3p: 1-арил-3-(3,5-диоксо-4-азатетрацикло[5.3.2.0^2,6.0^8,10]додец-
11-ен-4-ил)мочевины — Selivanov et al., Pharm Chem J 51:439 (2017),
10.1007/s11094-017-1629-9 (НИОЧ + ГНЦ ВБ «Вектор»; в абстракте —
«...studied for antiviral activity against smallpox vaccine virus», 3d
(4-толил) и 3h (3-фторфенил) «showed pronounced activity» — у них лучшие
EC50 серии, 0.56 и 0.51 мкг/мл). Значения в µg/mL с цензурой «>100»
(10 CC50 серии → ровно 100.0 после обратного пересчёта), организаторы
пересчитывают µg/mL → µM по MW и берут SI = CC50(µM)/EC50(µM) — якоря
train 3e (93.38/93.38 = 1.0), 3i (292.96/2.04 = 143.608),
3k (279.49/0.92 = 303.793) бит-в-бит.

Ряды '3','4','5','6','7','8' — ароилгидразиды того же имидного ядра из
второй статьи (та же конвенция µg/mL: '5'/'8' → CC50 ровно 100.0 —
цензура). EC50 подтверждены обзором JMC 2023 (10.1021/acs.jmedchem.3c00069,
док CHEMBL5346934): '3' 1.7, '4' 1.05, '5' 0.38, '6' 0.95, '8' 1.28 мкг/мл
— после пересчёта по MW совпадают с хранимыми парами бит-в-бит
(1.7×1000/464.6 = 3.66 и т.д.). Якорь train '4' (13.32/2.18 = 6.11).

Серия sid160xxxxx — 1-ацил-4-циклоалкил/арилсемикарбазиды НПВП (Perković
et al., EJMC 2012, 10.1016/j.ejmech.2012.02.046): «None of the compounds
were inhibitory to a broad panel of DNA and RNA viruses» — CC50 есть, EC50
нет → сентинелы y=10000 (конвенция организаторов, 10/10 train-строк серии).

Запуск: uv run python experiments/grishak_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULE = "grishak2017"
PMC = "10.1007/s11094-017-1629-9"

# (Title, smiles, si, sentinel)
TABLE = [
    # --- мочевины 3a-3p (PCJ 2017; SI = round(CC50_uM/EC50_uM, 3)) ---
    ("3a", "O=C(NC1=CC=CC=C1)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@@H]4C[C@H]34)[C@@H]2C1=O",
     106.275, False),
    ("3b", "CC1=C(NC(=O)NN2C(=O)[C@@H]3[C@@H]4C=C[C@@H]([C@H]5C[C@@H]45)[C@@H]3C2=O)C=CC=C1",
     86.921, False),
    ("3c", "CC1=CC=CC(NC(=O)NN2C(=O)[C@@H]3[C@@H]4C=C[C@@H]([C@H]5C[C@@H]45)[C@@H]3C2=O)=C1",
     26.115, False),
    ("3d", "CC1=CC=C(NC(=O)NN2C(=O)[C@@H]3[C@@H]4C=C[C@@H]([C@H]5C[C@@H]45)[C@@H]3C2=O)C=C1",
     529.286, False),
    ("3e", "COC1=C(NC(=O)NN2C(=O)[C@@H]3[C@@H]4C=C[C@@H]([C@H]5C[C@@H]45)[C@@H]3C2=O)C=CC=C1",
     1.0, False),
    ("3f", "COC1=CC=CC(NC(=O)NN2C(=O)[C@@H]3[C@@H]4C=C[C@@H]([C@H]5C[C@@H]45)[C@@H]3C2=O)=C1",
     1.0, False),
    ("3g", "COC1=CC=C(NC(=O)NN2C(=O)[C@@H]3[C@@H]4C=C[C@@H]([C@H]5C[C@@H]45)[C@@H]3C2=O)C=C1",
     10.626, False),
    ("3h", "O=C(NC1=CC(F)=CC=C1)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O",
     574.431, False),
    ("3i", "O=C(NC1=CC=C(F)C=C1)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O",
     143.608, False),
    ("3j", "O=C(NC1=C(Cl)C=CC=C1)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O",
     168.265, False),
    ("3k", "O=C(NC1=CC(Cl)=CC=C1)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O",
     303.793, False),
    ("3l", "O=C(NC1=CC(Cl)=CC(Cl)=C1)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O",
     1.0, False),
    ("3n", "CC1=CC=CC(C)=C1NC(=O)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O",
     30.566, False),
    ("3o", "O=C(NC1=CC=C(C(F)(F)F)C=C1)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O",
     7.692, False),
    ("3p", "CCOC(=O)C1=CC=C(NC(=O)NN2C(=O)[C@@H]3[C@@H]4C=C[C@@H]([C@H]5C[C@@H]45)[C@@H]3C2=O)C=C1",
     5.195, False),
    # --- ароилгидразиды '3'-'8' (та же конвенция µg/mL) ---
    ("3", "CCCCCCCCCCOC1=CC=C(C(=O)NN2C(=O)[C@@H]3[C@@H]4C=C[C@@H]([C@H]5C[C@@H]45)[C@@H]3C2=O)C=C1",
     9.191, False),
    ("4", "O=C(NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O)C1=C(O)C(Br)=CC(Br)=C1",
     6.11, False),
    ("5", "O=C(NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O)C1=CC([N+](=O)[O-])=C(O)C=C1",
     262.864, False),
    ("6", "O=C(CNC1=CC(Cl)=CC(Cl)=C1)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O",
     19.585, False),
    ("7", "CC1=CC=CC(C)=C1NC(=O)C(=O)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O",
     76.923, False),
    ("8", "CC1=CC=CC(C)=C1NC(=O)C(=O)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@@H]4C[C@H]34)[C@@H]2C1=O",
     78.207, False),
]

#: sid160xxxxx (EJMC 2012 семикарбазиды): CC50 есть, EC50 нет → сентинел
SID160 = {
    "test": ["sid160693662", "sid160704385", "sid160682846", "sid160699042",
             "sid160699041", "sid160704386", "sid160672205", "sid160672206",
             "sid160682847", "sid160693661", "sid160672204", "sid160682843",
             "sid160693663", "sid160704387", "sid160693664", "sid160682844",
             "sid160666887", "sid160672208", "sid160666884"],
    "train": ["sid160666886", "sid160666885", "sid160677530", "sid160677532",
              "sid160677534", "sid160688197", "sid160672207", "sid160704384",
              "sid160699040", "sid160677531"],
}


def _si_by_title(df: pd.DataFrame) -> dict:
    """(Title, smiles) → round(CC50_uM/IC50_uM, 3) по хранимым парам."""
    out = {}
    for _, r in df.iterrows():
        cc, ic = r.get("CC50_uM"), r.get("IC50_uM_VACV")
        if pd.isna(cc) or pd.isna(ic):
            continue
        out[(str(r["Title"]), r["SMILES"])] = round(float(cc) / float(ic), 3)
    return out


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    stored = _si_by_title(train)

    rows = []
    for title, smiles, si, sent in TABLE:
        # train-строки с этим (Title, smiles): значение обязано совпасть с y
        if ((train["Title"].astype(str) == title) & (train["SMILES"] == smiles)).any():
            y = float(train[(train["Title"].astype(str) == title) &
                            (train["SMILES"] == smiles)]["SI VACV"].iloc[0])
            assert y == si or abs(si - y) <= 1e-9 * max(abs(si), abs(y)), \
                f"'{title}': y={y} != {si}"
        assert ((test["Title"].astype(str) == title) & (test["SMILES"] == smiles)).any() or \
               ((train["Title"].astype(str) == title) & (train["SMILES"] == smiles)).any(), \
            f"'{title}': строки нет в данных"
        src = PMC
        rows.append({"Title": title, "sentinel": sent, "si": si, "smiles": smiles,
                     "pmc": src, "label": title, "rule": RULE})

    pmc_ejmc = "10.1016/j.ejmech.2012.02.046"
    for split, df in (("test", test), ("train", train)):
        for label in SID160[split]:
            sub = df[df["Title"].astype(str) == label]
            assert len(sub) == 1, f"{label}: {len(sub)} строк"
            r = sub.iloc[0]
            assert pd.isna(r["IC50_uM_VACV"]), f"{label}: IC50 не NaN"
            if split == "train":
                assert float(r["SI VACV"]) == 10000.0, f"{label}: y != 10000"
            rows.append({"Title": label, "sentinel": True, "si": 10000.0,
                         "smiles": r["SMILES"], "pmc": pmc_ejmc,
                         "label": label, "rule": RULE})

    out = pd.DataFrame(rows)
    assert not out.duplicated(["smiles", "Title"]).any()
    path = ROOT / "data" / "processed" / "grishak_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк "
          f"(мочевины/гидразиды {len(TABLE)}, сентинелов sid160 "
          f"{out['sentinel'].sum()})")


if __name__ == "__main__":
    main()
