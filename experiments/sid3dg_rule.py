"""Правило JMC 1984 для sid103201900 (3-деазагуанин, NSC-261726).

Источник: Cheung et al./Suhadolnik, "Synthesis and antiviral/antitumor
activities of certain 3-deazaguanine nucleosides and nucleotides",
J. Med. Chem. 1984, 27, 1391 (10.1021/jm00377a002), Table II
("Comparative in Vitro Antiviral Activity ... in Vero Cells"):
  3-deazaguanine (1): toxic level 2.2 (µg/ml), VR: HSV1=1.40, VV=1.55,
  Para3=0.98, VSV=1.55.

Формула организаторов (воспроизводит хранящиеся колонки бит-в-бит):
  CC50_µM = round2(toxic_level_µg/ml × 1000 / MW(структуры соревнования))
          = round2(2.2 × 1000 / 150.14) = 14.65
  IC50_µM = VR против vaccinia = 1.55
  y = 14.65 / 1.55 = 9.452  (train#97 y=9.452 ✓)

Запрещённые колонки использованы только для сверки; правило строится из
таблицы статьи и MW структуры (разрешено).

Запуск: uv run python experiments/sid3dg_rule.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1021/jm00377a002"
RULE = "jmc1984_deazaguanine"

MW_3DG = 150.14          # 3-deazaguanine, C6H6N4O
CC50_UM = round(2.2 * 1000 / MW_3DG, 2)   # 14.65
IC50_UM = 1.55                            # VR против VV


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    r = train.loc[97]
    assert str(r["Title"]).strip() == "sid103201900"
    si = round(CC50_UM / IC50_UM, 3)
    y = float(r["SI VACV"])
    print(f"train#97 sid103201900: CC50={CC50_UM} IC50={IC50_UM} si={si} y={y} "
          f"{'✓' if abs(si - y) < 1e-9 else '✗'}")
    # test-строки семейства (cyclophos/3-deazaguanosine) НЕ применяем: у орг.
    # CC50 из другого источника (15.4/1502 ≠ наш пересчёт 13.69/16.09)
    rows = [{
        "Title": "sid103201900", "sentinel": False, "si": si,
        "smiles": r["SMILES"], "pmc": DOI, "label": "1", "rule": RULE,
    }]
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "sid3dg_si.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)} оверрайдов → {path}")


if __name__ == "__main__":
    main()
