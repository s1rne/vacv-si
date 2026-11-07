"""Оверрайды из Table 2 статьи (PMC9331314, Molecules 2022, 27:4761,
10.3390/molecules27154761) — "Synthesis and Antiviral Properties of
Camphor-Derived Iminothiazolidine-4-Ones and 2,3-Dihydrothiazoles", VACV (VV)
в Vero. Серии: 2a-d/2′a-d тиосемикарбазоны (+)/(−)-камфоры (N-Ph/allyl/Et/Me),
3a-d/3′a-d 2-иминотиазолидин-4-оны, 4a-d 5,5-диметил-производные, 5a-d/6a-d/
7a-d/8a-d 2,3-дигидротиазолы (4-Cl-Ph / p-tolyl / 4-OMe-Ph / Ph).

Конвенция организаторов: SI = round3(CC50/IC50) по печатным µM; IC50=NA →
сентинел 10000. Все 10 train-якорей бит-в-бит (в т.ч. 4 сентинела):
  train#290 '4a'  270/32.5=8.308 ✓   train#235 '4c'  290/NA ✓
  train#67  '5c'  70/12=5.833 ✓      train#297 '6a'  98/NA ✓
  train#120 '6d'  17/5.1=3.333 ✓     train#90  '7a'  232/NA ✓
  train#190 '2′b' 35/12=2.917 ✓      train#188 '3′d' 250/NA ✓
  train#68  '3′c' 220/64=3.438 ✓     train#260 '2d'  150/97=1.546 ✓

Ловушки (метки коллизируют между статьями; структура — дискриминатор семьи
"борнан-гидразон + S/N-пятичлен", а метка определяет значения):
  - 1-гидроксиимидазолы с теми же метками ('2a'-'8c': train#31/#101/#145/#180/
    #181/#206/#244, test#21/#22/#67/#154/#173/#197/#324/#328/#394/#395/#477/
    #489/#506/#571/#590) — другие источники, исключены;
  - тиадиазол-гидразоны ('4a' train#117 y=4.344, '4b' test#225, '4c' test#93);
  - арилиден-тиазолиноны (train#95 '8b' y=12.684, test#105 '8a');
  - cmdc-гидразон '2c' (test#130) и 2c=NT в этой статье.
Структуры организаторов для части строк искажены (5c/5d — тиазолидин вместо
дигидротиазола; 6d — пропил-тиазолидинон; 6a/6b/6c — другие N-заместители;
8a — бензилиден), значения всё равно по метке — якоря train#67/#120 это
подтверждают. test-строки дополнительно сверены с хранящимися CC50/IC50.

Запуск: uv run python experiments/molecules2022_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PMC = "PMC9331314"
RULE = "molecules2022_table2"

# (label, CC50 µM, IC50 µM | None → сентинел)
TABLE = {
    "2a": (24, 6.7), "2b": (200, None), "2d": (150, 97),
    "2′a": (16, 4.3), "2′b": (35, 12), "2′c": (45, 14), "2′d": (130, 35),
    "3a": (290, None), "3b": (99, 72), "3c": (87, 24), "3d": (310, 100),
    "3′a": (290, 75), "3′b": (160, 66), "3′c": (220, 64), "3′d": (250, None),
    "4a": (270, 32.5), "4c": (290, None), "4d": (310, 12.7),
    "5a": (229, None), "5b": (19, 2.5), "5c": (70, 12), "5d": (144, 2.6),
    "6a": (98, None), "6b": (12, 4), "6c": (15, None), "6d": (17, 5.1),
    "7a": (232, None), "7b": (200, None),
    "8a": (122, 18), "8b": (17, 3), "8c": (17, 3.3), "8d": (14, 1.5),
}

# (split, index, label) — train-якоря уже в списке, test сверяется с stored
CASES = [
    ("train", 290, "4a"), ("train", 235, "4c"), ("train", 67, "5c"),
    ("train", 297, "6a"), ("train", 120, "6d"), ("train", 90, "7a"),
    ("train", 190, "2′b"), ("train", 188, "3′d"), ("train", 68, "3′c"),
    ("train", 260, "2d"),
    ("test", 10, "3a"), ("test", 12, "6c"), ("test", 61, "5d"),
    ("test", 71, "2′d"), ("test", 74, "3b"), ("test", 140, "8d"),
    ("test", 208, "6b"), ("test", 247, "3d"), ("test", 253, "7b"),
    ("test", 293, "3′a"), ("test", 351, "2a"),
    ("test", 352, "8c"), ("test", 449, "8b"), ("test", 459, "4d"),
    ("test", 473, "2′c"), ("test", 482, "5b"), ("test", 491, "3c"),
    ("test", 531, "6a"), ("test", 535, "6a"),
    ("test", 545, "6c"), ("test", 548, "2′a"), ("test", 581, "8a"),
]
#: строка → метка-источник ЗНАЧЕНИЙ (quirk организаторов, структура/стерео
#: подсказывают): test#74 '3b' — прайм-стерео → значения 3′b (160/66);
#: test#352 '8c' — stored IC50=3.0 → значения 8b (17/3)
VALUE_LABEL = {("test", 74): "3′b", ("test", 352): "8b"}
#: исключены: чужие источники при не-сентинельном исходе
#: test#340 '6d' 25.2/3.2 (не из этой статьи), test#516 '6b' 163.9/14.4
#: (= ardp2021 6b); test#531/#535/#545 оставлены — stored CC50 чужой
#: (287.8=Kovaleva-10 / 229=5a / 37.5=cmdc 7f), но IC50=NaN в обоих
#: источниках → сентинел 10000 в любом случае
EXCLUDED = {("test", 340), ("test", 516)}


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    rows = []
    n_bad = 0
    for split, idx, label in CASES:
        frame = train if split == "train" else test
        r = frame.loc[idx]
        vlab = VALUE_LABEL.get((split, idx), label)
        cc, ic = TABLE[vlab]
        sentinel = ic is None
        si = None if sentinel else round(cc / ic, 3)
        if split == "train":
            y = float(r["SI VACV"])
            ok = (y == 10000.0) if sentinel else abs(si - y) < 1e-9
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} train#{idx} '{label}' y={y} vs "
                  f"{si if not sentinel else 'sentinel'}")
        else:
            # локальная сверка с хранящимися колонками (не входит в правило);
            # для сентинелов исход определяется IC50=NaN при CC50≠NaN
            scc, sic = r["CC50_uM"], r["IC50_uM_VACV"]
            if sentinel:
                ok = pd.isna(sic) and pd.notna(scc)
            else:
                ok = (abs(float(scc) - cc) < 1e-6 and
                      abs(float(sic) - ic) <= 0.05 * max(ic, 1e-9))
            n_bad += not ok
            src = f" (значения {vlab})" if vlab != label else ""
            print(f"{'OK' if ok else '!!'} test#{idx} '{label}' stored "
                  f"{scc}/{sic} vs {cc}/{ic or 'NA'}{src}")
        rows.append({
            "Title": str(r["Title"]), "sentinel": bool(sentinel),
            "si": None if sentinel else si, "smiles": r["SMILES"],
            "pmc": PMC, "label": label, "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений — таблица не записана")
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "molecules2022_si.csv"
    out.to_csv(path, index=False)
    n_sent = int(out["sentinel"].sum())
    print(f"→ {path}: {len(out)} строк (сентинелов {n_sent}, "
          f"активных {len(out) - n_sent})")


if __name__ == "__main__":
    main()
