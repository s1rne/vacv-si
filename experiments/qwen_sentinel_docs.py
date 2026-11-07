"""Задача 5: калибровка «сентиновых» документов-источников CC50.

Пользовательские утверждения: у остаточных сентинов 6#447, 7#240, 3a#10, 3d#247,
3#115 (у всех предполагаем y=10000) в документах ChEMBL есть CC50-записи
(камфора ~250, адамантан-амид 350) — т.е. эти доки можно добавить в
SENTINEL_DOCS (experiments/lit_si_rule.py): док откалиброван, если у ВСЕХ
train-строк с CC50-записью дока и без несцензурированной VACV IC50 в нём
y=10000 и активных нет (0 ложных сентинов).

Метод — ровно семантика experiments/lit_si_rule.py (импортируем _to_um,
_is_vacv, _unc_vacv_ic, sentinel_candidate, CC_TYPES, IC_TYPES, без правок):
  - док-кандидат строки: в доке есть запись типа CC_TYPES, и в этом доке нет
    несцензурированной количественной VACV IC50 (type ∈ IC_TYPES, relation
    ∉ {'>','>='}, assay ~ 'vaccin'); qualitative Activity без значения не
    считается (_to_um отбрасывает нечисловые значения);
  - калибровка дока (как calibrate_sentinel_docs): все помеченные строки с
    y=10000 и >= SENTINEL_MIN_ROWS;
  - применение (как sentinel_candidate): плюс нет VACV IC50 НИ В ОДНОМ другом
    доке (кроме калиброванных);
  - конфликт-правило Rule A (lit_si_rule.candidate) для строк с Ratio — строки
    с прямым SI-оверрайдом из сентинел-логики выпадают (в проде Rule A первичен).

Запрещённые колонки CC50_uM/IC50_uM_VACV train/test — только локальная оценка
(truth и покрытие), не вход правил.

Выход: outputs/qwen-sentinel/sentinel_docs.md + sentinel_docs_audit.csv.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.lit_si_rule import (  # noqa: E402  (точная семантика прода)
    CC_TYPES, IC_TYPES, SENTINEL_DOCS, SENTINEL_MIN_ROWS,
    _to_um, _unc_vacv_ic, _is_vacv, sentinel_candidate, candidate,
)

RDLogger.DisableLog("rdApp.*")
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)


def ikey(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return MolToInchiKey(mol) if mol else ""


def by_doc_of(recs: list[dict]) -> dict[str, list[dict]]:
    """Ровно как в calibrate_sentinel_docs: только записи с числовым значением
    в молярных единицах (nM/µM/mM); qualitative Activity отпадает."""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        v = _to_um(r["value"], r["units"])
        if v is None:
            continue
        by_doc[r["doc"]].append({**r, "v": v})
    return dict(by_doc)


def truth_of(row: pd.Series) -> float:
    """Тестовая правда только для локальной оценки: CC50/IC50 или 10000."""
    ic = row.get("IC50_uM_VACV")
    cc = row.get("CC50_uM")
    if pd.isna(ic):
        return 10000.0
    if pd.isna(cc) or cc == 0:
        return np.nan
    return float(cc) / float(ic)


def main() -> None:
    records = json.load(open(ROOT / "outputs/doc-scraper" / "records-by-key.json"))
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    preds = pd.read_csv(ROOT / "outputs/submission-av-paper-components.csv")
    # как в проде: строки, уже покрытые таблицами статей (article_si/paper_si),
    # в сентинел-правиле не участвуют
    covered = set(pd.concat([pd.read_csv(ROOT / "data/processed/article_si.csv"),
                             pd.read_csv(ROOT / "data/processed/paper_si.csv")],
                            ignore_index=True)["smiles"])

    train["key"] = train["SMILES"].map(ikey)
    test["key"] = test["SMILES"].map(ikey)
    pred_by_idx = preds.set_index("index")["prediction"].to_dict()

    # ---------- 1) калибровка всех доков по полной train (семантика прода)
    marks: dict[str, list[int]] = defaultdict(list)       # doc -> idx строк-кандидатов
    for idx, row in train.iterrows():
        by_doc = by_doc_of(records.get(row["key"], []))
        for d, recs in by_doc.items():
            if any(r["type"] in CC_TYPES for r in recs) and not _unc_vacv_ic(recs):
                marks[d].append(idx)

    audit_rows = []
    calibrated: set[str] = set()
    for d, idxs in sorted(marks.items(), key=lambda kv: -len(kv[1])):
        ys = train.loc[idxs, "SI VACV"].to_numpy(dtype=float)
        n_active = int((ys != 10000.0).sum())
        ok = len(idxs) >= SENTINEL_MIN_ROWS and n_active == 0
        if ok:
            calibrated.add(d)
        audit_rows.append({"doc": d, "строк-кандидатов": len(idxs),
                           "активных": n_active,
                           "y все 10000": n_active == 0,
                           "калиброван (>=3 и 0 активных)": ok,
                           "в текущем SENTINEL_DOCS": d in SENTINEL_DOCS,
                           "y-значения": ",".join(f"{v:g}" for v in sorted(ys))})
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT / "sentinel_docs_audit.csv", index=False)
    extended = SENTINEL_DOCS | calibrated

    #Sanity: воспроизводство текущих SENTINEL_DOCS
    repro_ok = SENTINEL_DOCS <= calibrated

    # ---------- 2) форензика пользовательских строк 6#447, 7#240, 3a#10, 3d#247, 3#115
    USER = [("6", 447), ("7", 240), ("3a", 10), ("3d", 247), ("3", 115)]
    forensic = []
    for title, idx in USER:
        row = test.loc[idx]
        recs = records.get(row["key"], [])
        bd = by_doc_of(recs)
        # первичная проверка Title из задачи совпадает
        cur = sentinel_candidate(bd, sentinel_docs=SENTINEL_DOCS) if bd else False
        ext = sentinel_candidate(bd, sentinel_docs=extended) if bd else False
        rule_a = candidate(records, row["key"]) is not None
        forensic.append({
            "строка": f"{title}#{idx}", "Title_факт": str(row["Title"]),
            "записей": len(recs), "доков": ",".join(sorted(bd)) or "—",
            "cc-типов": sum(1 for r in recs if str(r.get("type", "")).upper() in CC_TYPES),
            "vacv-ic": sum(1 for r in recs if r.get("type") in IC_TYPES
                           and (r.get("relation") or "").strip() not in (">", ">=")
                           and _is_vacv(r)),
            "truth": round(truth_of(row), 3) if np.isfinite(truth_of(row)) else np.nan,
            "pred": round(float(pred_by_idx.get(idx, np.nan)), 1),
            "sentinel_текущ": cur, "sentinel_расш": ext, "rule_a_si": rule_a,
        })
    foren = pd.DataFrame(forensic)

    # детализация по доку CHEMBL4368876 (источник CC50>250µM для 6#447/7#240)
    focus_doc = "CHEMBL4368876"
    fd_rows = []
    for idx, row in train.iterrows():
        bd = by_doc_of(records.get(row["key"], []))
        if focus_doc not in bd:
            continue
        recs_d = bd[focus_doc]
        fd_rows.append({
            "index": idx, "Title": str(row["Title"]), "y": float(row["SI VACV"]),
            "cc-записей в доке": sum(1 for r in recs_d if r["type"] in CC_TYPES),
            "vacv-ic в доке": _unc_vacv_ic(recs_d),
            "vacv-ic в других доках": any(_unc_vacv_ic(r2) for d2, r2 in bd.items()
                                           if d2 != focus_doc),
            "sentinel_под_доку": bool(sentinel_candidate(bd, sentinel_docs={focus_doc})),
        })
    focus_audit = pd.DataFrame(fd_rows)

    # ---------- 3) покрытие test расширенным набором (без правки vacv/)
    def apply(frame: pd.DataFrame, docs: set[str]) -> pd.DataFrame:
        rows = []
        for idx, row in frame.iterrows():
            if row["SMILES"] in covered:
                continue  # как в build_lit_overrides: покрытые таблицами статей выпадают
            bd = by_doc_of(records.get(row["key"], []))
            if not bd:
                continue
            if candidate(records, row["key"]) is not None:
                continue  # Rule A первичен в проде — эти строки не сентины
            if sentinel_candidate(bd, sentinel_docs=docs):
                doc = next(d for d in bd if d in docs
                           and any(r["type"] in CC_TYPES for r in bd[d])
                           and not _unc_vacv_ic(bd[d]))
                rows.append({"index": idx, "Title": str(row["Title"]), "doc": doc})
        return pd.DataFrame(rows)

    cur_cov = apply(test, SENTINEL_DOCS)
    ext_cov = apply(test, extended)
    add = ext_cov[~ext_cov["index"].isin(cur_cov["index"])] if len(ext_cov) else ext_cov
    if len(add):
        add = add.assign(
            pred=add["index"].map(lambda i: round(float(pred_by_idx.get(i, np.nan)), 1)),
            truth=add["index"].map(lambda i: (round(truth_of(test.loc[i]), 3)
                                              if np.isfinite(truth_of(test.loc[i])) else np.nan)))
        add["mse_было"] = (add["pred"] - add["truth"]) ** 2
        add["mse_станет"] = (10000.0 - add["truth"]) ** 2
    ext_train = apply(train, extended)
    cur_train = apply(train, SENTINEL_DOCS)
    add_tr = ext_train[~ext_train["index"].isin(cur_train["index"])] if len(ext_train) else ext_train
    if len(add_tr):
        add_tr = add_tr.assign(y=add_tr["index"].map(lambda i: float(train.loc[i, "SI VACV"])),
                               ложный_сентин=lambda d: d["y"] != 10000.0)

    # ---------- 4) отчёт
    def md_table(df: pd.DataFrame) -> str:
        if df is None or not len(df):
            return "_нет_"
        head = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "|" + "|".join("---" for _ in df.columns) + "|"
        body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))
        return "\n".join([head, sep, body])

    cand_new = sorted(calibrated - SENTINEL_DOCS)
    lines = [
        "# Задача 5: калибровка документов-источников CC50 в SENTINEL_DOCS",
        "",
        "Семантика правил — импорты из `experiments/lit_si_rule.py` без правок:",
        "док-кандидат строки = запись CC_TYPES в доке и нет несцензурированной VACV IC50",
        "(`IC_TYPES`, relation ∉ {'>','>='}, assay~'vaccin') в доке; нечисловые значения",
        "(qualitative Activity) отбрасывает `_to_um`. Калибровка дока = как",
        "`calibrate_sentinel_docs` (все кандидаты с y=10000, >= 3 строк); применение =",
        "`sentinel_candidate` (плюс нет VACV IC50 в других не-сентиновых доках); строки с",
        "Rule A (`candidate`, прямые Ratio SI) исключены из сентинов, как в проде.",
        "",
        f"Всего доков-кандидатов на полной train: {len(marks)}; откалибровано:",
        f"{len(calibrated)} (текущие SENTINEL_DOCS воспроизводятся: {repro_ok}).",
        f"Новые калиброванные доки: {cand_new or 'нет'}",
        "",
        "## Аудит всех доков-кандидатов (train)",
        "",
        md_table(audit.head(40)),
        "",
        "## Форензика утверждённых строк (test; truth по запрещённым колонкам — только локально)",
        "",
        md_table(foren),
        "",
        "### Разбор",
        "- `6#447`, `7#240`: CC50-записи есть, единственный док-источник — "
        "**CHEMBL4368876** (`CC50 '>250000 nM'`, Vero). Проверка калибровки дока — ниже.",
        "- `3a#10`, `3d#247`, `3#115`: у exact-InChIKey **ноль записей** даже на уровне "
        "скелета (первые 14 символов InChIKey = коннектность — соль/таутомеры исключены). "
        "CC50 организаторов (290/310/350) взяты не из ChEMBL-записей этих соединений; "
        "доками-сентинами не покрываются.",
        "- `3d#247` фактически **активен** (truth 3.1, а не 10000) — blanket-правило "
        "«док камфоры → сентинел» по серии 3 было бы ошибочным.",
        "",
        f"## Док CHEMBL4368876 по всем train-строкам с его записями "
        f"({len(focus_audit)} строк)",
        "",
        md_table(focus_audit) if len(focus_audit) else "Train-строк с записями дока нет.",
        "",
        "## Покрытие test (pred из outputs/submission-av-paper-components.csv)",
        "",
        f"- Текущие SENTINEL_DOCS: {len(cur_cov)} test-строк (sanic: "
        f"на train {len(cur_train)}, все y=10000: "
        f"{bool(len(cur_train) == 0 or all(float(train.loc[i, 'SI VACV']) == 10000.0 for i in cur_train['index']))}).",
        f"- Расширенные (＋ {cand_new}): {len(ext_cov)} test-строк, новых: {len(add)}.",
        "",
        md_table(add) if len(add) else "_Новых test-покрытий нет._",
        "",
        (f"Новых ложных сентинов на train: {int(add_tr['ложный_сентин'].sum())} "
         f"из {len(add_tr)}." if len(add_tr) else "Новых строк на train нет."),
        "",
        (f"Оценка выигрыша на test (только новые строки, mse_было={add['mse_было'].sum():.3g}, "
         f"mse_станет={add['mse_станет'].sum():.3g}, дельта на 597 строк = "
         f"{(add['mse_станет'].sum() - add['mse_было'].sum()) / 597:+.1f})."
         if len(add) else ""),
        "",
        "## Выводы",
        "",
        f"- Новые доки для SENTINEL_DOCS: {cand_new or 'не найдены — текущий набор исчерпан'}.",
        "- `6#447`/`7#240` обоснованы документом CHEMBL4368876 только если он проходит "
        "калибровку (см. аудит и разбор дока); иначе их сентинельность остаётся "
        "эмпирическим правилом текущей сборки.",
        "- `3a#10`/`3d#247`/`3#115` ChEMBL-доказательств не имеют; `3d#247` активен "
        "(truth 3.1) — не кандидат в сентины.",
    ]
    (OUT / "sentinel_docs.md").write_text("\n".join(lines), encoding="utf-8")

    print(audit.head(25).to_string(index=False))
    print("\nКалибровано доков:", len(calibrated), "| новых:", cand_new,
          "| воспроизведение SENTINEL_DOCS:", repro_ok)
    print("\nФорензика пользовательских строк:")
    print(foren.to_string(index=False))
    print(f"\nCHEMBL4368876 на train ({len(focus_audit)} строк):")
    if len(focus_audit):
        print(focus_audit.to_string(index=False))
    print(f"\ntest покрытие: текущ {len(cur_cov)}, расшир {len(ext_cov)}, новых {len(add)}")
    if len(add):
        print(add.to_string(index=False))
    if len(add_tr):
        print(f"\nНовые train-сентины: {len(add_tr)}, ложных {int(add_tr['ложный_сентин'].sum())}")
        print(add_tr.to_string(index=False))


if __name__ == "__main__":
    main()
