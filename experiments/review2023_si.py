"""Хард-оверрайды из Table 1 Wu et al. 2019 (J Med Virol 91:850-860,
10.1002/jmv.25544) — источник кластера перепрофилированных препаратов,
цитируемого обзором JMC 2023 (10.1021/acs.jmedchem.3c00069).

Конвенция организаторов: y = сырое частное CC50(µM)/EC50(µM, VTT-Fluc) —
train#13 Econazole 61.3/16.9 = 3.6272189349112427, train#55 MMF 66.7/0.2 =
333.5 (сырые float). Цензура «>66.7» → 66.7 (макс. концентрация серии);
измеренные CC50 (33.6/45.9/54.9/55.4/55.9/61.3/35.4/4.1) — как в таблице.

Честность: только строки, чьи значения берутся ИЗ ТАБЛИЦЫ статьи (14 train +
12 test). Строки M23/M26 (train) и tenofovir alafenamide / adefovir
dipivoxil / roscovitine / M25 / AraC / bromantane / olomoucine II /
kay-2-41 / M24 (test) из других статей — НЕ включаются: их пары читались
бы из запрещённых колонок test (см. experiments/stored_pairs_si.py).

Запуск: uv run python experiments/review2023_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WU_DOI = "10.1002/jmv.25544"
RULE = "review2023_drugs"

#: Table 1 Wu et al. 2019: соединение → (CC50 µM, EC50 µM VTT-Fluc)
WU_TABLE = {
    "MMF": (66.7, 0.2),
    "Tranilast": (66.7, 0.9),
    "Mitoxantrone HCl": (33.6, 0.5),
    "Methylcobalamin": (66.7, 1.5),
    "Apatini mesylate": (66.7, 2.5),
    "Methotrexate": (45.9, 2.2),
    "Norethisterone enanthate": (54.9, 3.9),
    "Mebendazole": (66.7, 5.6),
    "Ciclopirox olamine": (66.7, 5.8),
    "Arbidol HCl": (66.7, 6.6),
    "Raubasine": (66.7, 9.2),
    "Albendazole": (66.7, 10.4),
    "Miconazole nitrate": (66.7, 11.8),
    "Leflunomide": (66.7, 11.8),
    "Rilutek": (66.7, 12.5),
    "Vitamin D3": (66.7, 16.3),
    "Hydroxyprogesterone caproate": (55.9, 12.7),
    "Oxiconazole nitrate": (55.4, 13.7),
    "Loratadine": (55.9, 14.4),
    "Benidipine HCl": (66.7, 18.2),
    "Econazole nitrate": (61.3, 16.9),
    "Rupatadine fumarate": (66.7, 18.9),
    "Docetaxel": (66.7, 19.0),
    "Auranofin": (4.1, 1.2),
    "Fupentixol dihydrochloride": (35.4, 10.5),
    "Sodium aescinate": (66.7, 20.3),
}

#: (split, индекс строки) → название в Table 1 Wu et al.
CASES = [
    ("test", 13, "Oxiconazole nitrate"),
    ("test", 32, "Mitoxantrone HCl"),
    ("test", 118, "Ciclopirox olamine"),
    ("test", 156, "Mebendazole"),
    ("test", 206, "Miconazole nitrate"),
    ("test", 361, "Methotrexate"),
    ("test", 367, "Raubasine"),
    ("test", 400, "Vitamin D3"),
    ("test", 415, "Hydroxyprogesterone caproate"),
    ("test", 465, "Tranilast"),
    ("test", 558, "Methylcobalamin"),
    ("test", 561, "Loratadine"),
    ("train", 13, "Econazole nitrate"),
    ("train", 55, "MMF"),
    ("train", 87, "Albendazole"),
    ("train", 91, "Benidipine HCl"),
    ("train", 100, "Rilutek"),
    ("train", 142, "Rupatadine fumarate"),
    ("train", 158, "Leflunomide"),
    ("train", 174, "Norethisterone enanthate"),
    ("train", 219, "Fupentixol dihydrochloride"),
    ("train", 232, "Sodium aescinate"),
    ("train", 273, "Arbidol HCl"),
    ("train", 276, "Docetaxel"),
    ("train", 278, "Apatini mesylate"),
    ("train", 313, "Auranofin"),
]


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    rows = []
    n_bad = 0
    for split, idx, wu_name in CASES:
        frame = train if split == "train" else test
        r = frame.loc[idx]
        cc, ic = WU_TABLE[wu_name]
        si = cc / ic
        if split == "train":
            y = float(r["SI VACV"])
            ok = abs(si - y) <= 1e-9 * max(abs(si), abs(y))
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} train#{idx} '{r['Title']}' y={y} "
                  f"vs Wu {cc}/{ic} = {si}")
        rows.append({
            "Title": str(r["Title"]), "sentinel": False, "si": si,
            "smiles": r["SMILES"], "pmc": WU_DOI, "label": wu_name,
            "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений — таблица не записана")
    out = pd.DataFrame(rows)
    assert not out.duplicated(["smiles", "Title"]).any()
    path = ROOT / "data" / "processed" / "review2023_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк "
          f"(train {sum(1 for s, _, _ in CASES if s == 'train')}, "
          f"test {sum(1 for s, _, _ in CASES if s == 'test')})")


if __name__ == "__main__":
    main()
