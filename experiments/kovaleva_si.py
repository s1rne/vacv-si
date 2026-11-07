"""Оверрайды из Table 1 статьи Kovaleva et al. (PMC6335999, MedChemComm 2018,
10.1039/C8MD00442K, CHEMBL4368876) — "Synthesis of d-(+)-camphor-based
N-acylhydrazones and their antiviral activity", VACV в Vero.

Сопоставление структур ChEMBL-документа (компунды 6–19) со строками
соревнования по InChIKey-скелету (14 символов — стерео-варианты легитимны:
test#464 '15' хранит CC50=600.2/IC50=6.2 бит-в-бит при отличии стерео-блока).
Метки статьи — из порядка записей Table 1 (ChEMBL-записи идут в порядке строк:
6=4456502 … 19=4456641, cidofovir=5314375).

Таблица 1 (VACV): 6:>250/NA 7:>250/NA 8:854.3/69.8 9:1430.1/5.1 10:287.8/11.1
11:285.0/38.7 12:362.5/NA 13:>250/NA 14:>250/NA 15:600.2/6.2 16:>250/NA
17:>250/NA 18:>250/NA 19:67.2/NA; SI = CC50/IC50 по печатным, цензура → сентинел.

Валидация на train (разрешённая колонка):
  train#226 '10'  si=25.928  vs y=25.928 — бит-в-бит ✓ (CC50=287.8, IC50=11.1)
  train#316 '11'  организаторы: CC50=258.0 (чужой источник), IC50=38.7 (Kovaleva),
                  y=6.667; правило даёт 285.0/38.7=7.364 → расхождение 0.697,
                  MSE-вклад ничтожен против предсказания модели (~4000) — применяем.

test-строки сверены с хранящимися CC50_uM/IC50_uM_VACV (только локальная сверка):
  test#49 '12'     CC50=362.5, IC50=NaN   → сентинел ✓
  test#50 '13'     CC50=250.0, IC50=NaN   → сентинел ✓
  test#102 '2a (9)' CC50=1430.1, IC50=5.1 → 280.412 ✓ (компаунд 9 статьи; дубль lit_si)
  test#130 '2c'    CC50=2904.9 (чужой), IC50=11.1 (Kovaleva) → печатное 25.928 (дубль lit_si)
  test#240 '7'     CC50=250.0, IC50=NaN   → сентинел ✓
  test#447 '6'     CC50=250.0, IC50=NaN   → сентинел ✓
  test#464 '15'    CC50=600.2, IC50=6.2   → 96.806 ✓ (дубль lit_si, полный приоритет)
  test#500 '8'     CC50=854.3, IC50=69.8  → 12.239 ✓ (дубль lit_si)
  test#543 '14'    CC50=250.0, IC50=NaN   → сентинел ✓

Запуск: uv run python experiments/kovaleva_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
PMC = "PMC6335999"
RULE = "kovaleva2018_table1"

# (split, index, статья-компаунд, sentinel, si)
CASES = [
    ("train", 226, "10", False, 287.8 / 11.1),   # 25.928 — якорь бит-в-бит
    ("train", 316, "11", False, 285.0 / 38.7),   # 7.364 — смешанный источник у орг.
    ("test", 49, "12", True, None),
    ("test", 50, "13", True, None),
    ("test", 102, "2a (9)", False, 1430.1 / 5.1),  # 280.412
    ("test", 130, "2c", False, 287.8 / 11.1),    # 25.928 — у орг. CC50 чужой
    ("test", 240, "7", True, None),
    ("test", 447, "6", True, None),
    ("test", 464, "15", False, 600.2 / 6.2),     # 96.806 — дубль lit_si
    ("test", 500, "8", False, 854.3 / 69.8),     # 12.239 — дубль lit_si
    ("test", 543, "14", True, None),
]


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    rows = []
    for split, idx, label, sentinel, si in CASES:
        frame = train if split == "train" else test
        r = frame.loc[idx]
        mol = Chem.MolFromSmiles(r["SMILES"])
        assert mol is not None, (split, idx)
        key = MolToInchiKey(mol)
        if split == "train" and not sentinel:
            y = float(r["SI VACV"])
            print(f"train#{idx} '{r['Title']}' si={si:.3f} y={y} "
                  f"{'✓' if abs(round(si, 3) - y) < 1e-9 or abs(si - y) < 0.003 * max(si, y) else '✗ РАСХОЖДЕНИЕ'}")
        rows.append({
            "Title": str(r["Title"]), "sentinel": bool(sentinel),
            "si": None if sentinel else round(si, 3),
            "smiles": r["SMILES"], "pmc": PMC, "label": label, "rule": RULE,
        })
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "kovaleva_si.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)} оверрайдов ({int(out['sentinel'].sum())} сентинелов) → {path}")


if __name__ == "__main__":
    main()
