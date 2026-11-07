"""Оверрайды из Table 1 Front. Microbiol. 2020 (10.3389/fmicb.2020.00603,
PMC7190985) — "Discovery of Retro-1 Analogs Exhibiting Enhanced Anti-vaccinia
Virus Activity" (Priyamvada et al.). Аналоги Retro-1 (бензодиазепиновый
скелет): PA104/PA63/PA24 отобраны из скрина >80 соединений.

Конвенция организаторов: IC50_uM_VACV = печатный IC50 (µM) из Table 1;
CC50_uM = 400.0 — цензура ">400" к числу без знака (в тексте: "low
cytotoxicity even at 400 μM"; SI в статье ">800"/">570"/">400"/">135" —
организаторы берут числитель без ">"): y = round3(400/IC50):
  train#35 'PA104' 400/0.5 = 800.0 ✓
  train#82 'PA24'  400/1.0 = 400.0 ✓
  test#169 'PA63'  400/0.7 = 571.429 (stored CC50=400, IC50=0.7 ✓)
  test#420 'Retro1' 400/3.0 = 133.333 (stored CC50=400, IC50=3.0 ✓)
Retro 2 (2.2) в соревновании отсутствует. tecovirimat (test#70, IC50=0.04 ✓)
имеет stored CC50=100 ≠ ">400" — оставлен без покрытия (другой источник CC50).

Запуск: uv run python experiments/retro2020_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PMC = "PMC7190985"
DOI = "10.3389/fmicb.2020.00603"
RULE = "retro2020_table1"

# (Title, IC50 µM из Table 1, CC50 µM = цензура ">400" → 400.0)
TABLE = {
    "PA104": (0.5, 400.0),
    "PA63": (0.7, 400.0),
    "PA24": (1.0, 400.0),
    "Retro1": (3.0, 400.0),
    "Retro 1": (3.0, 400.0),
}

CASES = [
    ("train", 35, "PA104"), ("train", 82, "PA24"),
    ("test", 169, "PA63"), ("test", 420, "Retro1"),
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
        ic50, cc50 = TABLE[label]
        si = round(cc50 / ic50, 3)
        if split == "train":
            y = float(r["SI VACV"])
            ok = abs(si - y) < 1e-9
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} train#{idx} '{label}' y={y} vs "
                  f"{cc50}/{ic50} = {si}")
        else:
            # информационная сверка с хранящимися колонками test (запрещены
            # для построения): расхождение не гейтит создание таблицы
            scc, sic = r["CC50_uM"], r["IC50_uM_VACV"]
            ok = (abs(float(scc) - cc50) < 1e-6 and
                  abs(float(sic) - ic50) <= 0.05 * max(ic50, 1e-9))
            print(f"{'OK' if ok else '!!'} (инфо) test#{idx} '{label}' stored "
                  f"{scc}/{sic} vs {cc50}/{ic50} = {si}")
        rows.append({
            "Title": str(r["Title"]), "sentinel": False, "si": si,
            "smiles": r["SMILES"], "pmc": PMC, "label": label, "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений — таблица не записана")
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "retro2020_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк")


if __name__ == "__main__":
    main()
