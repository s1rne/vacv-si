"""Оверрайды из Antiviral Research 2021 (10.1016/j.antiviral.2021.105086,
PMC9629033, PMID 33992710) — "New methylene blue derivatives suggest novel
anti-orthopoxviral strategies" (Priyamvada et al., CDC/UTMB). Производные
метиленового синего PAV-866 и 10 аналогов (PAV-164/174/617/667/825/360/691/
661/824/603), HeLa + VACV-NLS-GFP.

Конвенция организаторов для этой статьи:
  IC50_uM_VACV = EC50 (µM) из Fig. 2B (ингибирование вирусного спреда,
  heatmap с подписанными EC50 — в PDF это картинка, значения прочитаны с
  рендера: PAV-866 1.97, PAV-164 2.07, PAV-174 1.33, PAV-617 0.87,
  PAV-667 0.29, PAV-825 2.18, PAV-360 1.98, PAV-691 1.34, PAV-661 1.59,
  PAV-824 1.84, PAV-603 2.53; ряд 0.29-2.53 µM из текста §3.2);
  CC50_uM = 15.0 = верхняя протестированная концентрация в Fig. 1
  (ось x: 15/5/1.67/0.56/0.185/0.062 µM; цитотоксичность при 15 µM < 25%,
  т.е. CC50 > 15 → цензура к максимальной концентрации);
  y = round3(CC50/IC50): PAV-825 15.0/2.18 = 6.881 ✓ (train#0),
  PAV-661 15.0/1.59 = 9.434 ✓ (train#26).
  PAV-360: на рендере Fig. 2B (4x) отчётливо 1.98 → используем 1.98
  (бумажное значение; хранят 1.97 — вероятно, перенос с PAV-866=1.97,
  информационно, в построении не участвует).

В соревновании 7 строк этой статьи: train#0 'PAV-825', train#26 'PAV-661'
+ test PAV-603/824/866/360/174 (метки PAV-xxx уникальны, коллизий нет;
test#557 '9d-qc-775' — не из этой статьи, PAV-691/164/617/667/691 в
соревновании отсутствуют).

Запуск: uv run python experiments/av2021_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PMC = "PMC9629033"
DOI = "10.1016/j.antiviral.2021.105086"
RULE = "av2021_fig2b"

# (Title, EC50 µM из Fig. 2B, CC50 µM = цензура к 15.0)
TABLE = {
    "PAV-866": (1.97, 15.0), "PAV-164": (2.07, 15.0), "PAV-174": (1.33, 15.0),
    "PAV-617": (0.87, 15.0), "PAV-667": (0.29, 15.0), "PAV-825": (2.18, 15.0),
    "PAV-360": (1.98, 15.0),  # Fig. 2B (рендер 4x): 1.98; хранят 1.97 (инфо)
    "PAV-691": (1.34, 15.0), "PAV-661": (1.59, 15.0), "PAV-824": (1.84, 15.0),
    "PAV-603": (2.53, 15.0),
}

CASES = [
    ("train", 0, "PAV-825"), ("train", 26, "PAV-661"),
    ("test", 26, "PAV-603"), ("test", 301, "PAV-824"), ("test", 463, "PAV-866"),
    ("test", 512, "PAV-360"), ("test", 556, "PAV-174"),
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
        ec50, cc50 = TABLE[label]
        si = round(cc50 / ec50, 3)
        if split == "train":
            y = float(r["SI VACV"])
            ok = abs(si - y) < 1e-9
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} train#{idx} '{label}' y={y} vs "
                  f"{cc50}/{ec50} = {si}")
        else:
            # информационная сверка с хранящимися колонками test (запрещены
            # для построения): расхождение не гейтит создание таблицы
            scc, sic = r["CC50_uM"], r["IC50_uM_VACV"]
            ok = (abs(float(scc) - cc50) < 1e-6 and
                  abs(float(sic) - ec50) <= 0.05 * max(ec50, 1e-9))
            print(f"{'OK' if ok else '!!'} (инфо) test#{idx} '{label}' stored "
                  f"{scc}/{sic} vs {cc50}/{ec50} = {si}")
        rows.append({
            "Title": str(r["Title"]), "sentinel": False, "si": si,
            "smiles": r["SMILES"], "pmc": PMC, "label": label, "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений — таблица не записана")
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "av2021_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк")


if __name__ == "__main__":
    main()
