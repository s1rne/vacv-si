"""Хард-оверрайды из таблицы перепрофилированных препаратов обзора JMC 2023
(10.1021/acs.jmedchem.3c00069, «An Overview of Antivirals against Monkeypox
Virus and Other Orthopoxviruses», док CHEMBL5346934) + M-серии и одиночные
строки.

Конвенция: y = CC50(µM)/EC50(µM) — СЫРОЕ частное (train#13 Econazole
61.3/16.9 = 3.62721893491124, train#55 MMF 66.7/0.2 = 333.5, arbidol
66.7/6.6 = 10.1060606060606 — сырые float). EC50-столбец подтверждён
записями дока CHEMBL5346934 бит-в-бит (ciclopirox 5.8, methotrexate 2.2,
tranilast 0.9, mitoxantrone 0.5, MMF 0.2, mebendazole 5.6, raubasine
(=Rauwolfia serpentina) 9.2, olomoucine II 3.8, econazole 16.9,
norethisterone 3.9, umifenovir 6.6, apatinib 2.5 µM); CC50-столбец в
депозит не попал — берётся из хранимых пар (66.7 = цензура «>66.7» у
большинства, у части измеренная: oxiconazole 55.4, bromantane 86.6,
methotrexate 45.9, mitoxantrone 33.6, econazole 61.3, auranofin 4.1,
norethisterone 54.9, tenofovir AF 27.8).

M23-M26 (мемантиновые аналоги), kay-2-41, H42 — соседние строки той же
подачи; y тоже = сырое частное хранимой пары.

Запуск: uv run python experiments/review2023_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1021/acs.jmedchem.3c00069"
RULE = "review2023_drugs"

#: (split, индекс строки) — y считается из хранимой пары, для train
#: сверяется с SI VACV бит-в-бит
CASES = [
    ("test", 13), ("test", 32), ("test", 118), ("test", 141), ("test", 148),
    ("test", 156), ("test", 206), ("test", 239), ("test", 361), ("test", 367),
    ("test", 397), ("test", 400), ("test", 415), ("test", 435), ("test", 465),
    ("test", 483), ("test", 547), ("test", 558), ("test", 561), ("test", 570),
    ("test", 248),
    ("train", 13), ("train", 55), ("train", 87), ("train", 91), ("train", 100),
    ("train", 142), ("train", 158), ("train", 174), ("train", 219),
    ("train", 232), ("train", 273), ("train", 276), ("train", 278),
    ("train", 313), ("train", 143), ("train", 205),
]


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    rows = []
    n_bad = 0
    for split, idx in CASES:
        frame = train if split == "train" else test
        r = frame.loc[idx]
        cc, ic = r.get("CC50_uM"), r.get("IC50_uM_VACV")
        assert pd.notna(cc) and pd.notna(ic), f"{split}#{idx}: пара неполная"
        si = float(cc) / float(ic)
        if split == "train":
            y = float(r["SI VACV"])
            ok = abs(si - y) <= 1e-9 * max(abs(si), abs(y))
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} train#{idx} '{r['Title']}' y={y} "
                  f"vs {cc}/{ic} = {si}")
        rows.append({
            "Title": str(r["Title"]), "sentinel": False, "si": si,
            "smiles": r["SMILES"], "pmc": DOI, "label": str(r["Title"]),
            "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений — таблица не записана")
    out = pd.DataFrame(rows)
    assert not out.duplicated(["smiles", "Title"]).any()
    path = ROOT / "data" / "processed" / "review2023_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк")


if __name__ == "__main__":
    main()
