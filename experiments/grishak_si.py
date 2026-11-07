"""Хард-оверрайды для серии N-арилмочевин камфоримида (3a-3p).

Selivanov et al., Pharm Chem J 51:439 (2017), 10.1007/s11094-017-1629-9
(НИОЧ + ГНЦ ВБ «Вектор»; в абстракте — «...studied for antiviral activity
against smallpox vaccine virus», 3d (4-толил) и 3h (3-фторфенил) «showed
pronounced activity»). Table 3: TC50/IC50 в µg/mL с цензурой «>100»
(и «>33»/«>80»/«>37» у 3e/3f/3l, где IC50 не достигнут) — организаторы
пересчитывают µg/mL → µM по MW (round2) и берут SI = CC50(µM)/EC50(µM).
Значения ниже = round3 частного round2(µg×1000/MW_RDKit) — воспроизводят
хранимые пары бит-в-бит на всех 15 соединениях (train-якоря 3e
93.38/93.38 = 1.0, 3i 292.96/2.04 = 143.608, 3k 279.49/0.92 = 303.793;
12 test-строк, напр. 3b 296.4/3.41, 3h 292.96/0.51, 3p 252.9/48.68).

Ряды ароилгидразидов '3'-'8' (вторая статья группы): включены только '5'
и '8' — EC50 из записей дока обзора JMC 2023 (CHEMBL5346934: 0.38 и
1.28 µg/mL → round2 µM по MW бит-в-бит) + цензура серии «>100» µg/mL
(100×1000/MW совпадает с хранимыми µM бит-в-бит: 270.75 и 263.56);
'3'/'4'/'6'/'7' с измеренными TC50 (15.6/6.4/18.6/90.5 µg/mL) НЕ
включены — вторая статья не найдена, а значения из хранимых пар —
утечка запрещённых колонок (см. experiments/stored_pairs_si.py).

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

# (Title, smiles, si, sentinel) — si = round3(round2(TC50_µg×1000/MW)/
# round2(IC50_µg×1000/MW)) по Table 3 статьи
TABLE = [
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
    # --- ароилгидразиды второй статьи (та же конвенция µg/mL): EC50 из
    # --- записей дока обзора JMC 2023 (CHEMBL5346934), TC50 = цензура
    # --- серии «>100» µg/mL (воспроизводится round2(100×1000/MW) бит-в-бит);
    # --- '3'/'4'/'6'/'7' с измеренными TC50 (15.6/6.4/18.6/90.5 µg/mL) —
    # --- без второй статьи не включаются
    ("5", "O=C(NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C[C@@H]34)[C@@H]2C1=O)C1=CC([N+](=O)[O-])=C(O)C=C1",
     262.864, False),
    ("8", "CC1=CC=CC(C)=C1NC(=O)C(=O)NN1C(=O)[C@@H]2[C@@H]3C=C[C@@H]([C@@H]4C[C@H]34)[C@@H]2C1=O",
     78.207, False),
]

#: sid160xxxxx (EJMC 2012 семикарбазиды): CC50 есть, EC50 нет → сентинел
SID160 = {
    "test": ["sid160693662", "sid160704385", "sid160682846", "sid160699042",
             "sid160699041", "sid160704386", "sid160672205", "sid160672206",
             "sid160682847", "sid160693661", "sid160672204", "sid160682843",
             "sid160693663", "sid160704387", "sid160693664", "sid160682844",
             "sid160666887", "sid160672208", "sid160666884",
             "sid160677529", "sid160704382"],
    "train": ["sid160666886", "sid160666885", "sid160677530", "sid160677532",
              "sid160677534", "sid160688197", "sid160672207", "sid160704384",
              "sid160699040", "sid160677531"],
}


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")

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
        rows.append({"Title": title, "sentinel": sent, "si": si, "smiles": smiles,
                     "pmc": PMC, "label": title, "rule": RULE})

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
          f"(мочевины 3a-3p {len(TABLE)}, сентинелов sid160 "
          f"{out['sentinel'].sum()})")


if __name__ == "__main__":
    main()
