"""Оверрайды из Table 1 Chem. Biodiversity 2018 (10.1002/cbdv.201800153,
Sokolova/Yarovaya/Bormotov/Shishkina/Salakhutdinov) — "Discovery of a new class
of inhibitors of Vaccinia virus based on (-)-borneol from Abies sibirica and
(+)-camphor". Борнил-эфиры/амиды ароматических N-гетероциклических кислот
(соединения 4-40), VACV в Vero (TC50/IC50, µM, среднее±SD).

Конвенция организаторов: CC50_uM/IC50_uM_VACV = печатные средние TC50/IC50
(без ±SD), y = round3(CC50/IC50); IC50 = NA → сентинел 10000. Метки честные
(номера соединений статьи), но bare-номера коллизируют с другими сериями
(Kovaleva-гидразоны, Molecules-2022 тиозолины, камфорные амины) — матч по
(Title == номер) + терпеновый скелет. train-строки — все bornane
(CC1(C)C2CCC1(C)CC2, строгий гард); test — bornane ('28' фенхан, '29' пинен/
нол: "morpholine derivatives of (+)-fenchol 28 and (-)-nopol 29 showed no
antiviral activity") — для test гард свободнее: бициклический терпен с
эфиром/амидом (структурная принадлежность подтверждена сверкой хранимых
CC50/IC50 с печатными).

Все 15 train-строк бит-в-бит (в т.ч. 7 сентинелов: 5/10/40/20/12/24/21 →
печатные TC50 совпали, IC50=NA ✓). 16 test-строк сверены с хранящимися
CC50/IC50 (фингерпринт Qwen, outputs/qwen-sentinel/bornyl-esters.md; скрипт
experiments/qwen_bornyl_cbdv.py). train#217 '7' (тот же SMILES, что train#33
'36') — ДРУГАЯ статья (камфорные амины, задача 12c): fallback lookup по
smiles для него отключён (title-коллизия → не подменяем).

Запуск: uv run python experiments/cbdv2018_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PMC = "10.1002/cbdv.201800153"
RULE = "cbdv2018_table1"
BORNANE = Chem.MolFromSmarts("CC1(C)C2CCC1(C)CC2")


def _terpene_ester_amide(mol) -> bool:
    """Свободный гард для test: бициклический терпен + эфир/амид."""
    if mol is None or mol.GetNumHeavyAtoms() < 14:
        return False
    ri = mol.GetRingInfo()
    if ri.NumRings() < 2:
        return False
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        for b, n in zip(a.GetBonds(), a.GetNeighbors()):
            if n.GetSymbol() in ("O", "N") and \
                    b.GetBondTypeAsDouble() == 1:
                for b2, n2 in zip(n.GetBonds(), n.GetNeighbors()):
                    if n2.GetSymbol() == "C" and n2 is not a and \
                            any(bb.GetBondTypeAsDouble() == 2 and
                                nn.GetSymbol() == "O"
                                for bb, nn in zip(n2.GetBonds(),
                                                  n2.GetNeighbors())):
                        return True
    return False

# (label, TC50 µM, IC50 µM | None → сентинел)
TABLE = {
    "4": (34.68, 20.19), "5": (35.70, None), "6": (474.79, 12.89),
    "7": (216.36, 42.33), "8": (272.85, 161.58), "9": (479.96, 120.71),
    "10": (118.48, None), "11": (179.27, None), "12": (81.74, None),
    "13": (1010.63, 17.88), "14": (168.61, 3.45), "15": (182.88, 7.95),
    "16": (16.48, None), "17": (53.76, 19.39), "18": (156.07, 64.87),
    "19": (103.38, None), "20": (117.86, None), "21": (104.96, None),
    "22": (72.66, 21.17), "23": (61.72, 19.07), "24": (21.05, None),
    "25": (71.04, None), "28": (229.61, 138.21), "29": (120.69, None),
    "34": (134.99, 2.50), "35": (88.42, 13.87), "36": (501.80, 12.55),
    "37": (473.36, 7.47), "38": (583.44, 109.35), "39": (743.39, 33.50),
    "40": (1625.16, None),
}

CASES = [
    ("train", 28, "5"), ("train", 30, "10"), ("train", 33, "36"),
    ("train", 74, "17"), ("train", 96, "40"), ("train", 125, "20"),
    ("train", 141, "9"), ("train", 155, "7"), ("train", 156, "12"),
    ("train", 215, "24"), ("train", 220, "34"), ("train", 255, "8"),
    ("train", 270, "13"), ("train", 284, "21"), ("train", 285, "35"),
    ("test", 46, "28"), ("test", 112, "25"), ("test", 131, "6"),
    ("test", 143, "16"), ("test", 159, "18"), ("test", 180, "19"),
    ("test", 189, "37"), ("test", 191, "14"), ("test", 371, "22"),
    ("test", 419, "39"), ("test", 431, "23"), ("test", 472, "29"),
    ("test", 474, "4"), ("test", 536, "11"), ("test", 542, "15"),
    ("test", 553, "38"),
]


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    rows = []
    n_bad = 0
    for split, idx, label in CASES:
        frame = train if split == "train" else test
        r = frame.loc[idx]
        assert str(r["Title"]).strip() == label, \
            f"{split}#{idx}: Title '{r['Title']}' != label '{label}'"
        mol = Chem.MolFromSmiles(str(r["SMILES"]))
        if split == "train":
            assert mol is not None and mol.HasSubstructMatch(BORNANE), \
                f"train#{idx}: нет bornane-скелета"
        else:
            assert _terpene_ester_amide(mol), \
                f"test#{idx}: не бициклический терпен-эфир/амид"
        cc, ic = TABLE[label]
        sentinel = ic is None
        si = None if sentinel else round(cc / ic, 3)
        if split == "train":
            y = float(r["SI VACV"])
            ok = (y == 10000.0) if sentinel else abs(si - y) < 1e-9
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} train#{idx} '{label}' y={y} vs "
                  f"{si if not sentinel else 'sentinel'}")
        else:
            # локальная сверка с хранящимися колонками (не входит в правило)
            scc, sic = r["CC50_uM"], r["IC50_uM_VACV"]
            if sentinel:
                ok = pd.isna(sic) and pd.notna(scc)
            else:
                ok = (abs(float(scc) - cc) < 1e-6 and
                      abs(float(sic) - ic) <= 0.05 * max(ic, 1e-9))
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} test#{idx} '{label}' stored "
                  f"{scc}/{sic} vs {cc}/{ic or 'NA'}")
        rows.append({
            "Title": str(r["Title"]), "sentinel": bool(sentinel),
            "si": None if sentinel else si, "smiles": r["SMILES"],
            "pmc": PMC, "label": label, "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений — таблица не записана")
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "cbdv2018_si.csv"
    out.to_csv(path, index=False)
    n_sent = int(out["sentinel"].sum())
    print(f"→ {path}: {len(out)} строк (сентинелов {n_sent}, "
          f"активных {len(out) - n_sent})")


if __name__ == "__main__":
    main()
