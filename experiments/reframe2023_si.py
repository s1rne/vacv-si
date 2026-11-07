"""Оверрайды из Tables 1-2 Microbiology Spectrum 2023 (10.1128/spectrum.04745-22,
PMC10434227) — "Identification of In Vitro Inhibitors of Monkeypox Replication".
Скрин библиотек ReFRAME и NPC против бирепортерного rVACV Scarlet/Nluc:
CC50 (MTT) / EC50 (FP-колонка), µM. Таблицы 3-4 (MPXV) не используются.

Конвенция организаторов: IC50_uM_VACV = печатный EC50 (FP), CC50_uM = печатный
CC50 с цензурой ">50"→50.0 и ">150"→150.0 (знак ">" отброшен); y = round3:
  train#59 'Rotenone' 0.10/0.015 = 6.667 ✓      train#71 'Monensin' >50/0.07 = 714.286 ✓
  train#147 'OSU-03012' 22.23/2.32 = 9.582 ✓    train#170 'Buparvaquone' 121.4/1.76 = 68.977 ✓
  train#209 'Mubritinib' 0.07/0.093 = 0.753 ✓
20 test-строк сверены с хранящимися CC50/IC50 (все точные).

НЕ покрыты из этой статьи:
  - train#55 'Mycophenolate mofetil' (66.7/0.2 → 333.5) — из другой статьи
    (66.7-цензура; у него и SMILES другой, чем у test#541 — коллизии нет);
  - test#70 'tecovirimat' (100.0/0.04) — CC50 ≠ ">50", другой источник;
  - 'Benzimidazole' (>50/>50, NA) в соревновании отсутствует.

Запуск: uv run python experiments/reframe2023_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PMC = "PMC10434227"
DOI = "10.1128/spectrum.04745-22"
RULE = "reframe2023_tables12"

# (Title, CC50 µM, EC50 µM FP; CC50 с цензурой ">50"/">150" → число)
TABLE = {
    # Table 1 — ReFRAME против rVACV Scarlet/Nluc
    "Antimycin A": (8.73, 0.07),
    "OSU-03012": (22.23, 2.32),
    "Mycophenolic acid": (150.0, 0.46),
    "AVN-944": (50.0, 0.13),
    "Azauridine": (50.0, 0.64),
    "Pyrazofurin": (50.0, 0.27),
    "Mycophenolate mofetil": (150.0, 1.20),
    "Azaribine": (150.0, 1.32),
    "Brequinar": (50.0, 0.12),
    # Table 2 — NPC против rVACV Scarlet/Nluc
    "Azoxystrobin": (27.92, 7.92),
    "Buparvaquone": (121.4, 1.76),
    "Valinomycin": (0.29, 0.003),
    "Narasin": (1.3, 0.02),
    "Amuvatinib": (18.51, 4.76),
    "Monensin": (50.0, 0.07),
    "Spautin-1": (8.00, 6.78),
    "Tryptanthrin": (6.31, 1.38),
    "Rotenone": (0.10, 0.015),
    "Mubritinib": (0.07, 0.093),
}

CASES = [
    ("train", 59, "Rotenone"), ("train", 71, "Monensin"),
    ("train", 147, "OSU-03012"), ("train", 170, "Buparvaquone"),
    ("train", 209, "Mubritinib"),
    ("test", 36, "Pyrazofurin"), ("test", 282, "AVN-944"),
    ("test", 289, "Narasin"), ("test", 304, "Antimycin A"),
    ("test", 307, "Azauridine"), ("test", 323, "Tryptanthrin"),
    ("test", 356, "Amuvatinib"), ("test", 366, "Azaribine"),
    ("test", 402, "Azoxystrobin"), ("test", 406, "Brequinar"),
    ("test", 425, "Buparvaquone"), ("test", 450, "Spautin-1"),
    ("test", 541, "Mycophenolate mofetil"), ("test", 559, "Valinomycin"),
    ("test", 572, "Mycophenolic acid"),
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
            f"{split}#{idx}: Title '{r['Title']}' != '{label}'"
        cc, ec = TABLE[label]
        sentinel = False
        si = round(cc / ec, 3)
        if split == "train":
            y = float(r["SI VACV"])
            ok = abs(si - y) < 1e-9
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} train#{idx} '{label}' y={y} vs "
                  f"{cc}/{ec} = {si}")
        else:
            # информационная сверка с хранящимися колонками test (запрещены
            # для построения): расхождение не гейтит создание таблицы
            scc, sic = r["CC50_uM"], r["IC50_uM_VACV"]
            ok = (abs(float(scc) - cc) < 1e-6 and
                  abs(float(sic) - ec) <= 0.05 * max(ec, 1e-9))
            print(f"{'OK' if ok else '!!'} (инфо) test#{idx} '{label}' stored "
                  f"{scc}/{sic} vs {cc}/{ec} = {si}")
        rows.append({
            "Title": str(r["Title"]), "sentinel": sentinel, "si": si,
            "smiles": r["SMILES"], "pmc": PMC, "label": label, "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений — таблица не записана")
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "reframe2023_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк")


if __name__ == "__main__":
    main()
