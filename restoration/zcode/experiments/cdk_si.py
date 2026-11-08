"""Хард-оверрайды CDK-ингибиторов: Holcakova et al., Antivir Chem Chemother
2010, 20:133-142 (10.3851/IMP1460, PMC2948526) — источник кластера
«ингибиторы CDK» из обзора JMC 2023 (ref 167).

Table 1 (EC50 µM, колонка Vaccinia virus): olomoucine II 3.8±1.3,
roscovitine >20. Table 2 (CC50 µM, HFFF): 24h оба >100 (цензура → 100).
Организаторы взяли пары со СДВИГОМ СТРОК таблицы: roscovitine ← 3.8
(в статье это olomoucine II), olomoucine II ← 20 (в статье это roscovitine,
цензура «>20» → 20). CC50 = 100 (цензура «>100», 24h HFFF) у обоих.
Две test-строки (#239, #483), HFF — совпадает с клеточной линией статьи.

Запуск: uv run python experiments/cdk_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULE = "cdk2010_holcakova"
DOI = "10.3851/IMP1460"

# Title → (CC50 µM, EC50 µM Vaccinia; EC50 со сдвигом строк — см. докстринг)
TABLE = {
    "Roscovitine": (100.0, 3.8),
    "Olomoucine II": (100.0, 20.0),
}

CASES = [("test", "Roscovitine"), ("test", "Olomoucine II")]


def main() -> None:
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    rows = []
    for split, label in CASES:
        match = test[test["Title"].astype(str) == label]
        assert len(match) == 1, f"{split} '{label}': {len(match)} строк"
        r = match.iloc[0]
        cc, ec = TABLE[label]
        si = cc / ec
        rows.append({
            "Title": label, "sentinel": False, "si": si,
            "smiles": r["SMILES"], "pmc": DOI, "label": label, "rule": RULE,
        })
    out = pd.DataFrame(rows)
    assert not out.duplicated(["smiles", "Title"]).any()
    path = ROOT / "data" / "processed" / "cdk_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк")


if __name__ == "__main__":
    main()
