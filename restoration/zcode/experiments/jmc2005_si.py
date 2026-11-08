"""Оверрайды из Table 1 J. Med. Chem. 2005, 48:4690 (10.1021/jm050291s, PMID
16000005, Janeza/Balzarini/Andrei/Snoeck/De Clercq/Robins) — ациклические
аналоги фуро[2,3-d]/пирроло[2,3-d]пиримидин-2(3H)-онов (3a-i фурил C4-C11,
4a-i пирроло C4-C11 + 4a H, 6a-f алкинилфурил C6-C11, 2 — 5-этинилурацил,
3a — родительский фуро без C6-заместителя). Ассаи в HEL; колонки VV (vaccinia)
и CC50 (цитостатик).

Конвенция организаторов, ПЕРЕВЕРИФИЦИРОВАНАЯ через ChEMBL (doc
CHEMBL1140321 = эта статья): IC50_uM_VACV = VV (EC50, assay CHEMBL833242),
CC50_uM = CC50 (assay CHEMBL825110; ">" отброшен). Все 13 test-строк
перевыведены из ChEMBL бит-в-бит; запрещённые колонки test (CC50_uM,
IC50_uM_VACV) скриптом не читаются вовсе. Метки test-строк
выведены из структур (длина цепи + фуро/пирроло); «квирк 4e» снят: ChEMBL
подтверждает CC50=119 µM (MCC=400) — это печатные значения, а не перенос.
Кластеры значений (CC50, VV) → si:
  (200,400)→0.5 [2,3a,3b,3c,4a,4b,4c]; (200,16)→12.5 [3d,3f,3h];
  (140,200)→0.7 [3e]; (100,200)→0.5 [3g]; (119,200)→0.595 [4e];
  (132,40)→3.3 [3i]; (200,80)→2.5 [4d]; (65,80)→0.812 [4f];
  (32,16)→2.0 [4h]; (41,40)→1.025 [4i]; (50,200)→0.25 [6a,6b,6c];
  (50,40)→1.25 [6d,6f]; (42,40)→1.05 [6e].
Все 11 train-якорей бит-в-бит; метки и значения всех строк выводятся из
структур (длина цепи 6a-f = C6-C11) и ChEMBL без y: train#157 (11C =
ундецинил = 6f) подтверждён ChEMBL (CHEMBL191279: CC50=50, VV=40 → 1.25) —
ранее принятое «искажение структуры» было ошибкой подсчёта цепи.
Исключён test#466 'sid103459771': хранит пару строки GCV (130/100), но
структура — сахарная кислота, не ганцикловир (коллизия ChEMBL) — не выводимо
из статьи для этой структуры, оставлено без покрытия.

Запуск: uv run python experiments/jmc2005_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1021/jm050291s"
RULE = "jmc2005_table1"

# (split, idx, label, CC50 µM, VV µM) — CC50/VV с цензурой ">" → число;
# значения train — из Table 1, значения test — из ChEMBL (doc CHEMBL1140321)
CASES = [
    ("train", 6, "3g", 100.0, 200.0),
    ("train", 20, "3f", 200.0, 16.0),
    ("train", 36, "4d", 200.0, 80.0),
    ("train", 139, "4b", 200.0, 400.0),
    ("train", 154, "4f", 65.0, 80.0),
    ("train", 157, "6f", 50.0, 40.0),   # 11C ундецинил = 6f; ChEMBL: 50/40 ✓
    ("train", 177, "6e", 42.0, 40.0),
    ("train", 237, "4h", 32.0, 16.0),
    ("train", 252, "6d", 50.0, 40.0),
    ("train", 271, "4a", 200.0, 400.0),
    ("train", 272, "6c", 50.0, 200.0),
    ("test", 7, "3e", 140.0, 200.0),     # 7C фурил
    ("test", 152, "4e", 119.0, 200.0),   # 7C пирроло; CC50=119 (ChEMBL)
    ("test", 181, "3b", 200.0, 400.0),   # 4C фурил
    ("test", 207, "6b", 50.0, 200.0),    # 7C гепт-1-инил
    ("test", 238, "2", 200.0, 400.0),    # 5-этинил-1-(2-гидроксиэтоксиметил)урацил
    ("test", 251, "4i", 41.0, 40.0),     # 11C пирроло
    ("test", 303, "3c", 200.0, 400.0),   # 5C фурил
    ("test", 331, "4c", 200.0, 400.0),   # 5C пирроло
    ("test", 339, "3a", 200.0, 400.0),   # фуро без C6-заместителя
    ("test", 372, "3d", 200.0, 16.0),    # 6C фурил
    ("test", 462, "6a", 50.0, 200.0),    # 6C гекс-1-инил
    ("test", 496, "3i", 132.0, 40.0),    # 11C фурил
    ("test", 539, "3h", 200.0, 16.0),    # 10C фурил
]
EXCLUDED = {466}  # test 'sid103459771' — структура сахара, значения строки GCV


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    rows = []
    n_bad = 0
    for split, idx, label, cc, vv in CASES:
        frame = train if split == "train" else test
        r = frame.loc[idx]
        assert str(r["Title"]).strip().startswith("sid"), \
            f"{split}#{idx}: Title '{r['Title']}' не sid"
        si = round(cc / vv, 3)
        if split == "train":
            # y — разрешённая целевая колонка train: якоря обязаны совпасть
            y = float(r["SI VACV"])
            ok = abs(si - y) < 1e-9
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} train#{idx} '{label}' y={y} vs "
                  f"{cc}/{vv} = {si}")
        # test-строки: значения уже перевыведены из ChEMBL (doc
        # CHEMBL1140321); запрещённые CC50_uM/IC50_uM_VACV не читаются вовсе
        rows.append({
            "Title": str(r["Title"]), "sentinel": False, "si": si,
            "smiles": r["SMILES"], "pmc": DOI, "label": label, "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений train — таблица не записана")
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "jmc2005_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк (исключён test#466)")


if __name__ == "__main__":
    main()
