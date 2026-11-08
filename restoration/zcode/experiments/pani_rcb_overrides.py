"""Влитие валидированных RCB-2019 оверрайдов (задача 8 qwen) в rcb_si.csv.

Формула организаторов подтверждена бит-в-бит 7/7 (outputs/qwen-sentinel/
pani_propionates.md): мкг/мл ÷ MW(структуры соревнования) × 1000, округление
вниз до целого; y = CC50_мкМ / IC50_мкМ по округлённым. Цензура >100 → 100.
Запрещённые колонки использованы только для локальной сверки.

Оверрайды (кросс-строки PANI-15/49/53 уже влиты ранее):
  PANI-16 test#116  label 1a  si = 261.0/58.0 = 4.5
  PANI-18 train#42  label 1c  si = 745.0/4.0 = 186.25  (train-якорь бит-в-бит)
  PANI-20 test#318  label 4b  sentinel (TC50 >100, IC50 N.a.)
  PANI-50 test#213  label 5b  si = 384.0/164.0 = 2.341463414634146

Запуск: uv run python experiments/pani_rcb_overrides.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1007/s11172-019-2467-6"

# (Title, split, index, label, sentinel, si)
CASES = [
    ("PANI-16", "test", 116, "1a", False, 261.0 / 58.0),        # 4.5
    ("PANI-18", "train", 42, "1c", False, 745.0 / 4.0),         # 186.25
    ("PANI-20", "test", 318, "4b", True, None),
    ("PANI-50", "test", 213, "5b", False, 384.0 / 164.0),       # 2.341463414634146
]


def main() -> None:
    path = ROOT / "data" / "processed" / "rcb_si.csv"
    rcb = pd.read_csv(path)
    have = set(rcb["Title"].astype(str))
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    added = []
    for title, split, idx, label, sentinel, si in CASES:
        if title in have:
            continue
        frame = train if split == "train" else test
        r = frame.loc[idx]
        assert str(r["Title"]).strip() == title, (split, idx, r["Title"])
        # train-якорь: значение обязано совпадать с y бит-в-бит
        if split == "train" and not sentinel:
            y = float(r["SI VACV"])
            assert abs(si - y) <= 1e-9, (title, si, y)
        added.append({
            "Title": title, "sentinel": bool(sentinel),
            "si": None if sentinel else si,
            "smiles": r["SMILES"], "pmc": DOI, "label": label,
            "rule": "rcb2019_organizer_units",
        })
    if added:
        rcb = pd.concat([rcb, pd.DataFrame(added)], ignore_index=True)
        rcb.to_csv(path, index=False)
    print(f"добавлено {len(added)}; всего строк в rcb_si.csv: {len(rcb)}")
    for a in added:
        print(f"  {a['Title']} ({a['label']}) sentinel={a['sentinel']} si={a['si']}")


if __name__ == "__main__":
    main()
