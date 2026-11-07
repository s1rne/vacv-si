"""Оверрайды из Table 1 письма J. Med. Chem. 2007, 50:1442 (10.1021/jm061484y,
PMID 17373899, Smee/Bray/Charslaw/Wesely/Sidwell) — N-ацил-имидные гидразиды
техногенной камеры (3,3a,4,4a,5,5a-гексагидро-2H-циклопента[c]пиррол-1-карбонил),
прямой ряд тековиримата (ST-246 = соединение 14). Колонки Table 1: VV EC50 µM
(vaccinia), CC50 µM (цитотоксичность); цензуры организаторов: ">x" → x
(CC50 ">100(0%)" → 100, VV ">20" → 20 — подтверждено train#200 '5' y=5.0).

Конвенция организаторов: y = CC50/VV — для этого семейства СЫРОЕ частное без
round3 (train#9: 100/2.3 = 43.4782608695652; train#79: 92/7.7 =
11.9480519480519), у части строк round3 (train#78 '6' y=135.135, #183 '11'
43.478, #196 '16' 45.455). В таблицу пишем сырое частное; расхождение с
round3-вариантом даёт ошибку ~1e-7 по MSE — пренебрежимо.

Все соединения привязаны ПО СТРУКТУРЕ: эталонные ряды (REF_ROWS) задают
канонический SMILES каждого соединения, каждый случай обязан канонизоваться
в эталон своего соединения. 15 train-якорей (cmpd6/7/11/16 — по 2 строки с
одинаковыми структурами и значениями). 18 test-строк, значения подтверждены
ChEMBL-депозитом дока CHEMBL1142975 (14B: #487=CHEMBL223485 >86/0.02,
#501=CHEMBL374020 >100/2.0, #407=CHEMBL220098 >100/>20, #495=CHEMBL373720
>100/15.8); запрещённые колонки test (CC50_uM, IC50_uM_VACV) скриптом не
читаются вовсе.

Исключены (другие серии, источник — отдельная статья, не найдена):
  - семикарбазиды ArNH-C(=O)-NH-N-имид с метками 3a-3p: train#12 '3m (ster?)',
    #197 '3i', #203 '3k', #280 '3e'; test#25 '3j', #29 '3o', #83 '3g',
    #88 '3n', #138 '3l', #173 '3a', #197 '3b', #221 '3f', #328 '3c', #350 '3h',
    #460 '3p', #489 '3d' (CC50 226-309 µM);
  - оксалилы test#374 '7', #455 '8'; бензиламид test#311 '6'; гидразиды с
    R вне Table 1: test#30 '5' (3-NO2-4-OH), #39 '3' (4-децилокси),
    train#314 '4' (2-OH-3,5-Br2);
  - train#51 '1' — амид (не имид), в Table 1 строки 1 нет.
Соединения Table 1, не найденные в соревновании: нет — покрыты все 2-18.

Запуск: uv run python experiments/jm2007_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1021/jm061484y"
RULE = "jm2007_table1"

# соединение → (CC50 µM, VV EC50 µM); цензура ">" → число
TABLE = {
    "2": (86, 0.02), "3": (100, 2.0), "4": (92, 7.7), "5": (100, 20),
    "6": (100, 0.74), "7": (100, 0.5), "8": (100, 3.0), "9": (100, 0.04),
    "10": (100, 0.02), "11": (100, 2.3), "12": (100, 0.05), "13": (100, 0.02),
    "14": (100, 0.04), "15": (100, 0.02), "16": (100, 2.2), "17": (100, 15.8),
    "18": (100, 7.1),
}

# эталонные ряды: (split, idx, соединение) — задают структуру соединения
REF_ROWS = [
    ("train", 79, "4"), ("train", 200, "5"), ("train", 78, "6"),
    ("train", 46, "7"), ("train", 85, "8"), ("train", 168, "9"),
    ("train", 9, "11"), ("train", 135, "12"), ("train", 264, "13"),
    ("train", 199, "15"), ("train", 112, "16"),
    ("test", 471, "2"), ("test", 481, "3"), ("test", 407, "5"),
    ("test", 271, "10"), ("test", 70, "14"), ("test", 33, "17"),
    ("test", 54, "18"),
]

# (split, idx, метка, соединение)
CASES = [
    ("train", 79, "44", "4"), ("train", 200, "5", "5"), ("train", 78, "6", "6"),
    ("train", 129, "46", "6"), ("train", 46, "47", "7"), ("train", 300, "7", "7"),
    ("train", 85, "48", "8"), ("train", 168, "49", "9"), ("train", 9, "51", "11"),
    ("train", 183, "11", "11"), ("train", 135, "52", "12"), ("train", 264, "53", "13"),
    ("train", 199, "54", "15"), ("train", 112, "55", "16"), ("train", 196, "16", "16"),
    ("test", 471, "2", "2"), ("test", 487, "42", "2"), ("test", 481, "3", "3"),
    ("test", 501, "43", "3"), ("test", 551, "4", "4"), ("test", 407, "45", "5"),
    ("test", 294, "8", "8"), ("test", 320, "9", "9"), ("test", 271, "10", "10"),
    ("test", 329, "50", "10"), ("test", 297, "12", "12"), ("test", 62, "13", "13"),
    ("test", 70, "tecovirimat", "14"), ("test", 100, "14", "14"),
    ("test", 33, "17", "17"), ("test", 495, "56", "17"), ("test", 54, "57", "18"),
    ("test", 306, "18", "18"),
]


def main() -> None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchiKey

    RDLogger.DisableLog("rdApp.*")

    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")

    def canon(split: str, idx: int) -> str:
        # InChIKey: таутомеры пиразола (test#54 '57' / #306 '18') дают один ключ
        frame = train if split == "train" else test
        m = Chem.MolFromSmiles(frame.loc[idx, "SMILES"])
        assert m is not None, f"{split}#{idx}: SMILES не парсится"
        return MolToInchiKey(m)

    # эталонные структуры соединений
    ref: dict[str, str] = {}
    for split, idx, cmpd in REF_ROWS:
        c = canon(split, idx)
        prev = ref.get(c)
        assert prev in (None, cmpd), f"эталон {c}: {prev} и {cmpd}"
        ref[c] = cmpd
    assert len({v for v in ref.values()}) == len(TABLE), \
        "эталонами покрыты не все соединения TABLE"

    rows = []
    n_bad = 0
    for split, idx, label, cmpd in CASES:
        frame = train if split == "train" else test
        r = frame.loc[idx]
        c = canon(split, idx)
        assert ref.get(c) == cmpd, \
            f"{split}#{idx} '{label}': структура канонизуется в " \
            f"{ref.get(c)}, а не cmpd{cmpd}"
        cc, vv = TABLE[cmpd]
        si = cc / vv
        if split == "train":
            # y — разрешённая целевая колонка train: якорь обязан совпасть
            # с сырым частным (относительный допуск 1e-9: CSV хранит 15 знаков)
            # или его round3 (организаторы пишут и так, и так)
            y = float(r["SI VACV"])
            ok = abs(si - y) <= 1e-9 * max(abs(si), abs(y), 1e-12) \
                or y == round(si, 3)
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} train#{idx} '{label}' cmpd{cmpd} "
                  f"y={y} vs {cc}/{vv} = {si:.15g}")
        rows.append({
            "Title": str(r["Title"]), "sentinel": False, "si": si,
            "smiles": r["SMILES"], "pmc": DOI, "label": cmpd, "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений train — таблица не записана")
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "jm2007_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк "
          f"(train {sum(1 for s, *_ in CASES if s == 'train')}, "
          f"test {sum(1 for s, *_ in CASES if s == 'test')})")


if __name__ == "__main__":
    main()
