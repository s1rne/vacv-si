"""Задача 4: CC50-восстановление через MCC/TD50/ID50 — сколько строк presence
получит cytotox-доказательства, и подготовка патченой таблицы для CV.

Presence-признаки (data/processed/competition_presence.csv) строит
vacv/presence.py (cmd_presence в vacv/cli.py): cytotox-половина — записи с
measurement_class=="cytotox" по классификатору vacv/classify.py. Типы MCC и
ID50 не входят ни в CYTOTOX_TYPES, ни в AMBIGUOUS_TYPES → classify() даёт
"unknown" → curate.py выбрасывает их ("unknown_measurement_class"); TD50 в
CYTOTOX_TYPES входит, но проверим эмпириально — не теряется ли он по другим
фильтрам. Курательный CSV (data/interim/curated.csv.gz) локально отсутствует,
поэтому присутствие пересобираем из локального кэша всех activity-записей
outputs/doc-scraper/records-by-key.json (853 InChIKey, все типы), воспроизводя
семантику vacv/presence.summarise: медиана value_um по всем записям скелета,
censored = canonical_relation не в ("=", "~"), агрегация по inchikey14.

Конвертация массовых единиц (µg/mL и т.п.) — vacv.units.to_micromolar с
молекулярным весом RDKit; допуск значимости — MIN_UM..MAX_UM из vacv.units,
как в curate.py.

Запрещённые колонки train.csv (CC50_uM) — только оценка качества заполнения.

Выход: outputs/qwen-sentinel/mcc_restore.md, mcc_restore_rows.csv,
competition_presence_ext.csv (для CV).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import MolWt

from vacv.classify import CYTOTOX_TYPES
from vacv.units import MAX_UM, MIN_UM, canonical_relation, canonical_units, to_micromolar

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)

# Типы-кандидаты на включение в cytotox-половину (по заданию); переопределяется
# аргументом командной строки: --types MCC,ID50,TD50 --tag all
EXT_TYPES = {"MCC", "TD50", "ID50"}
TAG = "all"


def main() -> None:
    global EXT_TYPES, TAG
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default=",".join(sorted(EXT_TYPES)))
    ap.add_argument("--tag", default="all")
    args = ap.parse_args()
    EXT_TYPES = {t.strip().upper() for t in args.types.split(",") if t.strip()}
    TAG = args.tag
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))
    presence = pd.read_csv(ROOT / "data/processed/competition_presence.csv", low_memory=False)
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")

    from vacv.match_competition import load_split
    tr = load_split("train")
    te = load_split("test")
    skel = {}
    for frame in (tr, te):
        for _, row in frame.iterrows():
            if row.get("inchikey14"):
                skel.setdefault(row["inchikey14"], row["SMILES"])

    mw_by_smiles: dict[str, float] = {}
    def mw_of(smiles: str) -> float | None:
        if smiles not in mw_by_smiles:
            m = Chem.MolFromSmiles(smiles)
            mw_by_smiles[smiles] = MolWt(m) if m else None
        return mw_by_smiles[smiles]

    # --- скелеты → расширенные записи (только типы-кандидаты, конвертируемые)
    ext_by_skel: dict[str, list[dict]] = {}
    type_counts: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    for key, recs in records.items():
        skeleton = key[:14]
        if skeleton not in skel:
            continue
        for r in recs:
            t = str(r.get("type") or "").strip().upper()
            if t not in EXT_TYPES:
                continue
            type_counts[t] += 1
            mw = mw_of(skel[skeleton])
            v, reason = to_micromolar(r.get("value"), r.get("units"), mol_weight=mw)
            if v is None:
                skip_reasons[f"{t}:{reason}"] += 1
                continue
            if not (MIN_UM <= v <= MAX_UM):
                skip_reasons[f"{t}:outside_range"] += 1
                continue
            rel = canonical_relation(r.get("relation"))
            ext_by_skel.setdefault(skeleton, []).append({
                "type": t, "value_um": v, "censored": rel not in ("=", "~"),
                "units": canonical_units(r.get("units")), "doc": r.get("doc"),
                "assay": str(r.get("assay") or "")[:80],
            })

    # --- сколько записей уже учтено текущим presence (sanity по TD50)
    td50_leak = 0
    for skeleton, recs in ext_by_skel.items():
        if any(r["type"] == "TD50" for r in recs):
            row = presence[(presence["smiles"] == skel[skeleton])]
            if len(row) and not bool(row.iloc[0]["has_cytotox"]):
                td50_leak += 1

    # --- эффект на presence
    rows_patch = []
    patched = presence.copy()
    for skeleton, ext in ext_by_skel.items():
        rows = patched[patched["smiles"] == skel[skeleton]]
        if not len(rows):
            continue
        idxs = rows.index.to_numpy()
        first = rows.iloc[0]
        vals = np.array([r["value_um"] for r in ext])
        n_new = len(ext)
        best_new = (float(np.median(vals)) if pd.isna(first["best_cc50_um"])
                    else first["best_cc50_um"])
        # медиана по всем cytotox-записям: у строк без прежних записей — только новые
        patched.loc[idxs, "n_cytotox"] = int(first["n_cytotox"]) + n_new
        patched.loc[idxs, "has_cytotox"] = True
        patched.loc[idxs, "best_cc50_um"] = best_new
        if pd.isna(first["best_cc50_um"]):
            patched.loc[idxs, "cc50_censored_any"] = any(r["censored"] for r in ext)
            patched.loc[idxs, "cc50_uncensored_any"] = any(not r["censored"] for r in ext)
        rows_patch.append({
            "skeleton": skeleton, "smiles": skel[skeleton], "n_ext": n_new,
            "types": ",".join(sorted({r["type"] for r in ext})),
            "values_um": ",".join(f"{v:.3f}" for v in np.sort(vals)),
            "all_censored": all(r["censored"] for r in ext),
            "was_nan_cc50": bool(pd.isna(first["best_cc50_um"])),
            "was_no_cytotox": bool(not first["has_cytotox"]),
            "best_um_new": best_new,
        })
    patched_df = pd.DataFrame(rows_patch)

    # --- метрики по train/test (без двойного счёта дублей smiles в train)
    train_titles = tr["Title"].to_numpy()
    summary = []
    for split, frame in (("train", train), ("test", test)):
        sl = tr if split == "train" else te
        sub = presence[presence["split"] == split]
        n_nan = int(sub["best_cc50_um"].isna().sum())
        n_nocyto = int((~sub["has_cytotox"].astype(bool)).sum())
        uniq = {}
        for r in rows_patch:
            uniq.setdefault(r["smiles"], r)
        hit = [r for r in uniq.values() if r["smiles"] in set(sl["SMILES"])]
        fill = [r for r in hit if r["was_nan_cc50"]]
        flip = [r for r in hit if r["was_no_cytotox"]]
        summary.append({"split": split, "строк": len(sl),
                        "best_cc50_um NaN (сейчас)": n_nan,
                        "has_cytotox=False": n_nocyto,
                        "уникальных соединений задето": len(hit),
                        "заполнено best_cc50_um": len(fill),
                        "has_cytotox перевернётся": len(flip),
                        "все записи цензурированы": sum(1 for r in hit if r["all_censored"])})

    # --- оценка качества заполнения по запрещённым CC50_uM (только оценка)
    tr_idx = tr.set_index("SMILES")
    ratios = []
    eval_rows = []
    for r in rows_patch:
        smi = r["smiles"]
        if not r["was_nan_cc50"]:
            continue
        if smi not in tr_idx.index:
            continue
        ref = tr_idx.loc[smi, "CC50_uM"]
        ref = float(ref) if pd.notna(ref) else None
        if not ref:
            continue
        ratio = r["best_um_new"] / ref
        ratios.append(ratio)
        eval_rows.append({"smiles": smi, "заполнено_um": round(r["best_um_new"], 3),
                          "CC50_uM_train": ref, "отношение": round(ratio, 3),
                          "типы": r["types"], "цензура": r["all_censored"]})
    ratios = np.array(ratios, dtype=float)

    det = pd.DataFrame(eval_rows)
    pd.DataFrame(rows_patch).to_csv(OUT / f"mcc_restore_rows_{TAG}.csv", index=False)
    patched.to_csv(OUT / f"competition_presence_ext_{TAG}.csv", index=False)

    def md_table(df: pd.DataFrame) -> str:
        if not len(df):
            return "_нет_"
        head = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "|" + "|".join("---" for _ in df.columns) + "|"
        body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))
        return "\n".join([head, sep, body])

    summ = pd.DataFrame(summary)
    skip_df = pd.DataFrame(sorted(skip_reasons.items(), key=lambda x: -x[1]),
                           columns=["причина пропуска", "записей"])
    type_df = pd.DataFrame(sorted(type_counts.items(), key=lambda x: -x[1]),
                           columns=["тип", "записей (всего)"])

    lines = [
        "# Задача 4: CC50-восстановление через MCC/TD50/ID50",
        "",
        "Presence-признаки строит `vacv/presence.py` (`cmd_presence` в `vacv/cli.py`),",
        "cytotox-половина = записи с `measurement_class==\"cytotox\"` по `vacv/classify.py`.",
        "MCC и ID50 не входят ни в `CYTOTOX_TYPES`, ни в `AMBIGUOUS_TYPES` → `unknown` →",
        "`curate.py` их выбрасывает. TD50 в `CYTOTOX_TYPES` есть; эмпирически соединений с",
        f"записями TD50, где presence всё же has_cytotox=False: {td50_leak} (т.е. TD50",
        "в целом не теряется — теряются MCC и ID50).",
        "",
        "Курательный CSV (data/interim/curated.csv.gz) отсутствует, поэтому записи взяты из",
        "локального кэша всех activity `outputs/doc-scraper/records-by-key.json`, семантика",
        "агрегатов воспроизведена из `vacv/presence.summarise` (медиана по записям скелета,",
        "censored = canonical_relation ∉ {=, ~}), допуск значений — MIN_UM..MAX_UM из `vacv/units.py`.",
        "",
        "## Записи-кандидаты",
        "",
        md_table(type_df),
        "",
        "Пропущено при конвертации:",
        "",
        md_table(skip_df),
        "",
        "## Эффект на presence",
        "",
        md_table(summ),
        "",
        "## Качество заполнения (оценка по запрещённой колонке CC50_uM, только локально)",
        "",
        (f"Оцениваемых строк (train, CC50_uM известна): {len(ratios)}. "
         f"Медианное отношение заполненного к реальному CC50: "
         f"{np.median(ratios):.2f}; в пределах x2: "
         f"{np.mean((ratios > 0.5) & (ratios < 2)):.0%}." if len(ratios)
         else "Оцениваемых строк нет (у всех заполненных CC50_uM отсутствует в train.csv)."),
        "",
        md_table(det),
        "",
        "## Выводы",
        "",
        f"- Расширение cytotox-типов {sorted(EXT_TYPES)} даёт best_cc50_um для",
        f"  {summ.iloc[0]['заполнено best_cc50_um']} train-строк (было NaN {summ.iloc[0]['best_cc50_um NaN (сейчас)']})",
        f"  и {summ.iloc[1]['заполнено best_cc50_um']} test-строк; переворачивает has_cytotox у",
        f"  {summ.iloc[0]['has_cytotox перевернётся'] + summ.iloc[1]['has_cytotox перевернётся']} соединений.",
        "- Патченная таблица для CV: `outputs/qwen-sentinel/competition_presence_ext.csv`",
        "  (CV-обёртка: `experiments/qwen_cv_presence_ext.py --presence outputs/qwen-sentinel/competition_presence_ext_{TAG}.csv`).",
    ]
    (OUT / f"mcc_restore_{TAG}.md").write_text("\n".join(lines), encoding="utf-8")

    print(summ.to_string(index=False))
    print()
    print(type_df.to_string(index=False))
    print()
    print(skip_df.to_string(index=False))
    print()
    print(f"train оценка заполнения: n={len(ratios)}", end="")
    if len(ratios):
        print(f", медианное отношение={np.median(ratios):.2f}, в пределах x2={np.mean((ratios > 0.5) & (ratios < 2)):.0%}")
    else:
        print()
    print(det.to_string(index=False))


if __name__ == "__main__":
    main()
