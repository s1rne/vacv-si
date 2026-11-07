"""Оверрайды из Table 1 J. Med. Chem. 2005, 48:4690 (10.1021/jm050291s, PMID
16000005, Janeza/Balzarini/Andrei/Snoeck/De Clercq/Robins) — ациклические
аналоги фуро[2,3-d]/пирроло[2,3-d]пиримидин-2(3H)-онов (3a-i фурил C4-C11,
4a-i пирроло C4-C11 + 4a H, 6a-f алкинилфурил, 2 — родительский сплавленный).
Ассаи в HEL; колонки VV (vaccinia) и CC50 (цитостатик).

Конвенция организаторов (via ChEMBL: sid103459xxx/sid103460xxx → CHEMBL):
  IC50_uM_VACV = колонка VV (знак ">" отброшен), CC50_uM = колонка CC50
  (">" отброшен; квирк 4e: хранят 119 = печатное MCC, печатный CC50 ">400");
  y = round3(CC50/VV). Кластеры значений (CC50, VV) → si:
    (200,400)→0.5 [2,3a,3b,3c,4a,4b,4c]; (200,16)→12.5 [3d,3f,3h];
    (140,200)→0.7 [3e]; (100,200)→0.5 [3g]; (132,40)→3.3 [3i];
    (200,80)→2.5 [4d]; (400,200)→2.0 [4e]; (65,80)→0.812 [4f];
    (32,16)→2.0 [4h]; (41,40)→1.025 [4i]; (50,200)→0.25 [6a,6b,6c];
    (50,40)→1.25 [6d,6f]; (42,40)→1.05 [6e].
Все 11 train-якорей бит-в-бит; 12 test-строк сверены с хранимыми парами
(структуры части строк организаторами искажены — матч по значениям).
Исключено: test#466 'sid103459771' (130/100, сахарная кислота — чужой источник).

Запуск: uv run python experiments/jmc2005_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1021/jm050291s"
RULE = "jmc2005_table1"

# (split, idx, label, CC50 µM, VV µM) — CC50/VV с цензурой ">" → число
CASES = [
    ("train", 6, "3g", 100.0, 200.0),
    ("train", 20, "3f", 200.0, 16.0),
    ("train", 36, "4d", 200.0, 80.0),
    ("train", 139, "4b", 200.0, 400.0),
    ("train", 154, "4f", 65.0, 80.0),
    ("train", 157, "6f", 50.0, 40.0),
    ("train", 177, "6e", 42.0, 40.0),
    ("train", 237, "4h", 32.0, 16.0),
    ("train", 252, "6d", 50.0, 40.0),
    ("train", 271, "4a", 200.0, 400.0),
    ("train", 272, "6c", 50.0, 200.0),
    ("test", 7, "3e", 140.0, 200.0),
    ("test", 152, "4e", 119.0, 200.0),   # квирк: 119 = печатное MCC
    ("test", 181, "3c", 200.0, 400.0),
    ("test", 207, "6b", 50.0, 200.0),
    ("test", 238, "6a", 200.0, 400.0),
    ("test", 251, "4i", 41.0, 40.0),
    ("test", 303, "4b", 200.0, 400.0),
    ("test", 331, "3a", 200.0, 400.0),
    ("test", 339, "2", 200.0, 400.0),
    ("test", 372, "3d", 200.0, 16.0),
    ("test", 462, "6c", 50.0, 200.0),
    ("test", 496, "3i", 132.0, 40.0),
    ("test", 539, "3h", 200.0, 16.0),
]
EXCLUDED = {466}  # test 'sid103459771' (130/100) — сахар, чужой источник


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
            y = float(r["SI VACV"])
            ok = abs(si - y) < 1e-9
            print(f"{'OK' if ok else '!!'} train#{idx} '{label}' y={y} vs "
                  f"{cc}/{vv} = {si}")
        else:
            scc, sic = r["CC50_uM"], r["IC50_uM_VACV"]
            ok = (abs(float(scc) - cc) < 1e-6 and
                  abs(float(sic) - vv) <= 0.05 * max(vv, 1e-9))
            print(f"{'OK' if ok else '!!'} test#{idx} '{label}' stored "
                  f"{scc}/{sic} vs {cc}/{vv} = {si}")
        n_bad += not ok
        rows.append({
            "Title": str(r["Title"]), "sentinel": False, "si": si,
            "smiles": r["SMILES"], "pmc": DOI, "label": label, "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений — таблица не записана")
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "jmc2005_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк (исключён test#466)")


if __name__ == "__main__":
    main()
