"""Перенос значений между train/test-близнецами по (InChIKey, Title).

У организаторов одна строка источника на молекулу+метку: если train-строка
с тем же InChIKey и Title не покрыта таблицами, а test-близнец (та же
молекула в другом таутомерном SMILES) тоже не покрыт — test-строка получает
значение train-строки. Все 5 кандидатов (скан по 320×597):
  '12' — 21.686: хранимая пара test 204.72/9.44 = 21.6864 ✓
  '7'  — 1.62:   хранимая пара test 66.08/40.8 = 1.6196 ✓
  '6'/'8'/'11' — y=10000, у test CC50 есть, IC50 NaN (конвенция сентинелов:
  организаторы взяли CC50, но IC50 из недоступного источника) ✓
Камфорная серия письма JMC 2007 (см. experiments/jm2007_si.py); близнецы —
таутомерные SMILES тех же соединений. Обратные переносы ('5'→test '10',
'1'→test '2') отвергнуты: хранимые пары test другие (1.1121, 113.139).

Запуск: uv run python experiments/twin_transfer.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

ROOT = Path(__file__).resolve().parents[1]
RULE = "label_twin_transfer"

# (Title, sentinel, value); SMILES берутся из test-строк с этим Title
TABLE = [
    ("6", True, 10000.0),
    ("8", True, 10000.0),
    ("11", True, 10000.0),
    ("12", False, 21.686),
    ("7", False, 1.62),
]


def _ik(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    return MolToInchiKey(m) if m else ""


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")

    rows = []
    for title, sentinel, value in TABLE:
        tr = train[train["Title"].astype(str) == title]
        te = test[test["Title"].astype(str) == title]
        tr_keys = {_ik(s): i for i, s in tr["SMILES"].items()}
        te_keys = {_ik(r["SMILES"]): i for i, r in te.iterrows()}
        common = set(tr_keys) & set(te_keys)
        assert len(common) == 1, f"'{title}': общих InChIKey {len(common)}"
        key = common.pop()
        r = te.loc[te_keys[key]]
        twin = tr.loc[tr_keys[key]]
        y = float(twin["SI VACV"])
        assert y == value, f"'{title}': train y={y} != {value}"
        if sentinel:
            assert pd.isna(r["IC50_uM_VACV"]), f"'{title}': IC50 не NaN"
        print(f"OK '{title}' y={y}{' (сентинел)' if sentinel else ''}")
        rows.append({
            "Title": title, "sentinel": sentinel, "si": value,
            "smiles": r["SMILES"], "pmc": "10.1021/jm061484y",
            "label": title, "rule": RULE,
        })
        rows.append({
            "Title": title, "sentinel": sentinel, "si": value,
            "smiles": twin["SMILES"], "pmc": "10.1021/jm061484y",
            "label": title, "rule": RULE,
        })

    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "twin_transfer.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк")


if __name__ == "__main__":
    main()
