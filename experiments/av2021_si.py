"""Оверрайды из Antiviral Research 2021 (10.1016/j.antiviral.2021.105086,
PMC9629033, PMID 33992710) — "New methylene blue derivatives suggest novel
anti-orthopoxviral strategies" (Priyamvada et al., CDC/UTMB). Производные
метиленового синего PAV-866 и 10 аналогов (PAV-164/174/617/667/825/360/691/
661/824/603), HeLa + VACV-NLS-GFP.

Конвенция организаторов для этой статьи:
  IC50_uM_VACV = EC50 (µM) из Fig. 2B (ингибирование вирусного спреда,
  heatmap с подписанными EC50 — в PDF это картинка, значения прочитаны с
  рендера: PAV-825 = 2.18, PAV-661 = 1.59; ряд 0.29-2.53 µM из текста §3.2);
  CC50_uM = 15.0 = верхняя протестированная концентрация в Fig. 1
  (ось x: 15/5/1.67/0.56/0.185/0.062 µM; цитотоксичность при 15 µM < 25%,
  т.е. CC50 > 15 → цензура к максимальной концентрации);
  y = round3(CC50/IC50): PAV-825 15.0/2.18 = 6.881 ✓ (train#0),
  PAV-661 15.0/1.59 = 9.434 ✓ (train#26).

В соревновании только 2 строки этой статьи (train#0 'PAV-825', train#26
'PAV-661'; test-двойников по SMILES нет). Остальные аналоги (PAV-866 и др.)
в соревновании отсутствуют. Метки PAV-xxx уникальны (коллизий нет).

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
    "PAV-825": (2.18, 15.0),
    "PAV-661": (1.59, 15.0),
}

CASES = [("train", 0, "PAV-825"), ("train", 26, "PAV-661")]


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
        y = float(r["SI VACV"])
        ok = abs(si - y) < 1e-9
        n_bad += not ok
        print(f"{'OK' if ok else '!!'} {split}#{idx} '{label}' y={y} vs "
              f"{cc50}/{ec50} = {si}")
        # test-двойников по SMILES нет (проверено в авг-2026), но на всякий
        # случай добавим их при появлении
        twins = test[test["SMILES"] == r["SMILES"]]
        for ti, tr_row in twins.iterrows():
            print(f"  test-двойник #{ti} '{str(tr_row['Title']).strip()}'")
            rows.append({
                "Title": str(tr_row["Title"]), "sentinel": False, "si": si,
                "smiles": tr_row["SMILES"], "pmc": PMC, "label": label,
                "rule": RULE,
            })
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
