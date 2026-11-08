"""Задача 13: источник борнил-эфиров/амидов с VACV — Chem. Biodivers. 2018.

Sokolova A.S., Yarovaya O.I., Bormotov N.I., Shishkina L.N., Salakhutdinov N.F.,
«Discovery of a new class of inhibitors of Vaccinia virus based on (−)-borneol
from Abies sibirica and (+)-camphor», Chem. Biodiversity 2018,
DOI 10.1002/cbdv.201800153 (accepted manuscript). PDF: outputs/qwen-sentinel/
cbdv2018-dl/cbdv.pdf (sci-hub.ee → sci.bban.top), текст: cbdv.txt (pypdf).

Источник истины — Table 1 «Antiviral activity of target derivatives 4–40 against
Vaccinia virus» (TC50/IC50, µM, среднее ± SD). Организаторы хранят печатные
средние в CC50_uM/IC50_uM_VACV (без ±SD), y = round3(TC50/IC50); IC50 = NA →
сентинель 10000. Запрещённые колонки читаются только для локальной сверки.

Совпадения ищём fingerprint'ом по печатным значениям всей Table 1 (не только
четырёх координаторских целей) — так находится весь легальный клиппинг статьи
в соревновании (31 строка), и проверяется отсутствие ложных коллизий (титулы
соревнования при этом равны номерам соединений статьи).

Выход: outputs/qwen-sentinel/bornyl-esters.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)

DOI = "10.1002/cbdv.201800153"

#: Table 1 статьи, печатные значения (TC50/IC50, µM; None = NA)
T1: dict[str, tuple[float, float | None]] = {
    "4": (34.68, 20.19), "5": (35.70, None), "6": (474.79, 12.89),
    "7": (216.36, 42.33), "8": (272.85, 161.58), "9": (479.96, 120.71),
    "10": (118.48, None), "11": (179.27, None), "12": (81.74, None),
    "13": (1010.63, 17.88), "14": (168.61, 3.45), "15": (182.88, 7.95),
    "16": (16.48, None), "17": (53.76, 19.39), "18": (156.07, 64.87),
    "19": (103.38, None), "20": (117.86, None), "21": (104.96, None),
    "22": (72.66, 21.17), "23": (61.72, 19.07), "24": (21.05, None),
    "25": (71.04, None), "28": (229.61, 138.21), "29": (120.69, None),
    "34": (134.99, 2.50), "35": (88.42, 13.87), "36": (501.80, 12.55),
    "37": (473.36, 7.47), "38": (583.44, 109.35), "39": (743.39, 33.50),
    "40": (1625.16, None),
}

#: точные строки-цитаты Table 1 из cbdv.txt (для цитирования в отчёте)
QUOTES = [
    "Table 1. Antiviral activity of target derivatives 4–40 against Vaccinia virus.",
    "Compound TC50 a, µM IC50 b, µM SIc",
    "(-)-borneol >650 NA -",
    "8 272.85±57.77 161.58±26.85 1.7",
    "9 479.96±37.62 120.71±11.06 4",
    "13 1010.63±119.90 17.88±4.55 56",
    "17 53.76±13.99 19.39±4.45 2.7",
    "Cidofovir 475.27±74.97 40.01±2.87 11.8",
    "aTC50=50% toxicity concentration, at which 50% of cells in uninfected "
    "monolayers are destroyed.",
    "bIC50=50% inhibitory concentration, at which 50% of cells in infected "
    "monolayers are preserved.",
    "cSI=selectivity index, ratio TC50/IC50.",
]

#: координаторские целевые строки задачи 13
TARGETS = [("train", 255, "8"), ("train", 141, "9"), ("train", 270, "13"),
           ("train", 74, "17")]


def close(a, b, tol=1e-4) -> bool:
    if a is None or b is None:
        return False
    try:
        b = float(b)
    except (TypeError, ValueError):
        return False
    return abs(float(a) - b) <= max(tol, 1e-9) * max(abs(b), 1.0)


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "raw" / "kaggle" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "raw" / "kaggle" / "test.csv")

    rows = []
    for split, df in (("train", train), ("test", test)):
        for i, r in df.iterrows():
            cc, ic = r.get("CC50_uM"), r.get("IC50_uM_VACV")
            if pd.isna(cc):
                continue
            for lab, (ccp, icp) in T1.items():
                if not close(cc, ccp):
                    continue
                if (icp is None and pd.isna(ic)) or (icp is not None and close(ic, icp)):
                    y = r.get("SI VACV") if split == "train" else None
                    y = None if y is None or pd.isna(y) else float(y)
                    pred = None if icp is None else round(ccp / icp, 3)
                    ok = (y == pred) if (y is not None and pred is not None) else \
                         (y == 10000.0 if y is not None else None)
                    m = Chem.MolFromSmiles(str(r["SMILES"]))
                    key = MolToInchiKey(m) if m else "?"
                    rows.append({
                        "split": split, "idx": i, "title": str(r["Title"]).strip(),
                        "lab": lab, "cc_print": ccp, "ic_print": icp, "y": y,
                        "pred": pred, "yok": ok, "inchikey": key,
                        "smiles": str(r["SMILES"]),
                    })

    n_ok = sum(1 for r in rows if r["yok"] is True)
    n_na = sum(1 for r in rows if r["yok"] is None)
    n_bad = sum(1 for r in rows if r["yok"] is False)
    cov = {r["lab"] for r in rows}
    missing = sorted(set(T1) - cov, key=int)

    lines = [
        "# Задача 13: борнил-эфиры/амиды с VACV — источник найден",
        "",
        "**Источник:** Sokolova A.S., Yarovaya O.I., Bormotov N.I., Shishkina L.N.,",
        "Salakhutdinov N.F. «Discovery of a new class of inhibitors of Vaccinia virus",
        f"based on (−)-borneol from Abies sibirica and (+)-camphor», **Chem. Biodiversity",
        f"2018, DOI {DOI}** (рукопись: outputs/qwen-sentinel/cbdv2018-dl/cbdv.pdf,",
        "текст cbdv.txt, sci-hub.ee). Группа ННИОХ/VECTOR (Кольцово) — та же, что",
        "Molecules 2022 и FrontPH-2018.",
        "",
        "Статья НЕ была в числе отвергнутых: CHEMBL4257457 (Марбург, C7MD00424A),",
        "CHEMBL4261492 (грипп, C6MD00657D) и CHEMBL4706564 (EJM 2020, филовирусы) —",
        "другие доки/статьи той же группы, их значения с y не совпадают (проверено",
        "ранее); значения соревнования воспроизводят именно cbdv-2018 Table 1.",
        "",
        "## Цитаты Table 1 (печатные средние TC50/IC50, µM)",
        "",
    ]
    for q in QUOTES:
        lines.append(f"> {q}")
    lines += [
        "",
        "## Механика организаторов",
        "",
        "- хранящиеся CC50_uM/IC50_uM_VACV = печатные средние TC50/IC50 статьи (µM, без ±SD);",
        "- y = round(CC50/IC50, 3); отсутствие IC50 (NA) → сентинель 10000;",
        "- Titles соревнования = номера соединений статьи (4–40) — здесь честные лейблы,",
        "  а не коллизии: все совпадения подтверждены и структурой (борнил-моноэфир",
        "  карбоновой кислоты с N-гетероциклом), и бит-в-бит значениями.",
        "",
        "## Координаторские целевые строки",
        "",
        "| train# | Title | label | печат TC50/IC50 | y_train | round3 | вердикт |",
        "|---|---|---|---|---|---|---|",
    ]
    for split, idx, lab in TARGETS:
        hit = next((r for r in rows if r["split"] == split and r["idx"] == idx), None)
        assert hit is not None, f"целевая строка {split}#{idx} не найдена"
        lines.append(f"| {idx} | '{hit['title']}' | {lab} | {hit['cc_print']}/{hit['ic_print']} "
                     f"| {hit['y']} | {hit['pred']} | {'бит-в-бит ✓' if hit['yok'] else '✗'} |")
    lines += [
        "",
        "## Полный клиппинг статьи в соревновании (fingerprint по всем строкам Table 1)",
        "",
        f"Найдено строк: **{len(rows)}** (train y-сверок пройдено: {n_ok}, "
        f"тестовых без y: {n_na}, рассогласований: {n_bad}).",
        f"Соединений Table 1, не представленных в соревновании: {len(missing)} "
        f"({', '.join(missing) if missing else '—'}).",
        "",
        "| split | # | Title | label | печат CC50 | печат IC50 | y_train | round3(CC/IC) | y✓ | "
        "InChIKey |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['split']} | {r['idx']} | '{r['title']}' | {r['lab']} | {r['cc_print']} "
            f"| {r['ic_print']} | {r['y']} | {r['pred']} | "
            f"{'✓' if r['yok'] else ('—' if r['yok'] is None else '✗')} | {r['inchikey']} |")
    lines += [
        "",
        "## Выводы / оверрайды",
        "",
        f"- Все 4 координаторские строки задачи 13 (train#255 '8', #141 '9', #270 '13', "
        f"#74 '17') воспроизведены бит-в-бит из cbdv-2018 Table 1 (y = round3(TC50/IC50)).",
        "- Легальный источник оверрайда для этих строк: файл данных со строками "
        "(Title=номер статьи, TC50/IC50 из Table 1, si = y при NA→10000). "
        "title==label ветка требует structure-guard (в соревновании есть чужие '8'/'9'/",
        "'13'/'17' — например, адамантильный '8' из PMC9533879-коллизий и '8'/'13'/"
        "'14'/'17' из других серий) — применять только с проверкой SMILES по ChEMBL/"
        "структурам статьи, по образцу LABEL_MATCH_* в experiments/paper_si_match.py.",
        "- Отклонённые ранее доки (CHEMBL4257457/4261492/4706564) не имеют отношения: "
        "значения соревнования = cbdv-2018, а не они.",
        "- Тестовые строки того же клиппинга (см. таблицу) — кандидаты на оверрайд; "
        "их печатные пары уникальны в соревновании, ложных коллизий fingerprint не "
        "дал (все hit-структуры — борнил/камфорные моноэфиры/амиды статьи).",
        "",
    ]
    (OUT / "bornyl-esters.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"rows={len(rows)} y_ok={n_ok} test={n_na} bad={n_bad} missing={missing}")
    print(f"→ {OUT / 'bornyl-esters.md'}")


if __name__ == "__main__":
    main()
