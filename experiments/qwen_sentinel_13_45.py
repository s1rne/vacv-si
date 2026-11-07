"""Задача 10: сентинел-калибровка доков для train#173 '13' и train#230 '45' (y=10000).

Метод задачи 5 (семантика experiments/lit_si_rule.py без правок). Дополнительно к
штатной калибровке проверяется ОСЛАБЛЕННЫЙ вариант: док «CC50-only», где условие
«нет несцензурированной VACV IC50 в доке» снято (организаторы проигнорировали
ChEMBL-IC50 этих записей — CC50=12.9 бит-в-бит = запись дока CHEMBL4699424).
Калибровка ослабленного правила — только по fit=4/5 (эталон lit_si_cv), ложные
сентины на valid считаются явно; штатный (non-relaxed) вариант проверяется тоже.
Запрещённые колонки — только локальная сверка фактов (CC50_uM=50.7/12.9), в
правилах не участвуют (y = разрешённая SI VACV).

Выход: outputs/qwen-sentinel/sentinel_13_45.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from experiments.lit_si_rule import (  # noqa: E402
    CC_TYPES, SENTINEL_DOCS, SENTINEL_MIN_ROWS, _to_um, _unc_vacv_ic,
    _is_vacv, sentinel_candidate, candidate, calibrate_sentinel_docs,
    calibrate_pair_trusted,
)
from experiments.active_value2 import attach_nn
from experiments.lit_si_cv import apply_overrides, piece_mse, valid_lit_overrides
from experiments.stacking import fold_arrays
from experiments.stacking_eval import load_sources
from vacv.dataset import load
from vacv.literature import attach
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds

RDLogger.DisableLog("rdApp.*")
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [(173, "13", 50.7), (230, "45", 12.9)]


def md_table(df: pd.DataFrame) -> str:
    if df is None or not len(df):
        return "_нет_"
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = "\n".join("| " + " | ".join(map(str, r)) + " |"
                     for r in df.itertuples(index=False))
    return "\n".join([head, sep, body])


def ikey(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return MolToInchiKey(m) if m else ""


def by_doc_of(recs: list[dict]) -> dict[str, list[dict]]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        v = _to_um(r["value"], r["units"])
        if v is None:
            continue
        by_doc[r["doc"]].append({**r, "v": v})
    return dict(by_doc)


def relaxed_marks(doc: str, records: dict, key: str) -> bool:
    """Ослабленный док-кандидат: в доке есть CC_TYPES-запись; VACV IC50 в доке
    НЕ блокирует (гипотеза: организаторы её не видели/игнорировали)."""
    bd = by_doc_of(records.get(key, []))
    recs = bd.get(doc)
    return bool(recs) and any(str(r.get("type", "")).upper() in CC_TYPES for r in recs)


def main() -> None:
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    preds = pd.read_csv(ROOT / "outputs/submission-av-paper-components.csv")
    pred_by_idx = preds.set_index("index")["prediction"].to_dict()
    covered = set(pd.concat([pd.read_csv(ROOT / "data/processed/article_si.csv"),
                             pd.read_csv(ROOT / "data/processed/paper_si.csv")],
                            ignore_index=True)["smiles"])
    train["key"] = train["SMILES"].map(ikey)
    test["key"] = test["SMILES"].map(ikey)

    lines = ["# Задача 10: сентинел-доки для train#173 '13' и train#230 '45'", ""]

    # ---------- форензика целевых строк и их доков
    docs_of_interest: dict[str, dict] = {}
    for idx, title, cc_claim in TARGETS:
        row = train.loc[idx]
        assert str(row["Title"]).strip() == title
        key = row["key"]
        recs = records.get(key, [])
        bd = by_doc_of(recs)
        docs = sorted(bd)
        lines += [
            f"## train#{idx} Title='{title}' (y={row['SI VACV']})",
            "",
            f"- CC50 организаторов {cc_claim} µM; ChEMBL-доки точного ключа: {docs};",
            f"- записи: " + "; ".join(
                f"{r['type']} {r['value']} {r['units']} '{r['relation']}' @{r['doc']}"
                for r in recs) or "нет",
            f"- запись CC_TYPES ≈ {cc_claim} µM (допуск 5%) есть: "
            f"{any(str(r['type']).upper() in CC_TYPES and (v := _to_um(r['value'], r['units'])) is not None and abs(v - cc_claim) <= 0.05 * cc_claim for r in recs)}",
            "",
        ]
        for d in docs:
            docs_of_interest.setdefault(d, {"rows": []})
        # доки по скелету (на случай солевых ключей)
        sk = key[:14]
        for k in records:
            if k[:14] == sk:
                for d in by_doc_of(records[k]):
                    docs_of_interest.setdefault(d, {"rows": []})

    # ---------- все train/test строки каждого дока-кандидата
    for d in docs_of_interest:
        tr_rows, te_rows = [], []
        for idx, row in train.iterrows():
            if any(r.get("doc") == d for r in records.get(row["key"], [])):
                bd = by_doc_of(records.get(row["key"], []))
                tr_rows.append({"index": idx, "Title": str(row["Title"]),
                                "y": float(row["SI VACV"]),
                                "cc-в-доке": any(str(r["type"]).upper() in CC_TYPES
                                                 for r in bd.get(d, [])),
                                "vacv-ic-в-доке": _unc_vacv_ic(bd.get(d, [])),
                                "штатный-маркер": (any(str(r["type"]).upper() in CC_TYPES
                                                       for r in bd.get(d, []))
                                                   and not _unc_vacv_ic(bd.get(d, [])))})
        for idx, row in test.iterrows():
            if row["SMILES"] in covered:
                continue
            if any(r.get("doc") == d for r in records.get(row["key"], [])):
                te_rows.append({"index": idx, "Title": str(row["Title"]),
                                "pred": round(float(pred_by_idx.get(idx, np.nan)), 1)})
        docs_of_interest[d]["train"] = pd.DataFrame(tr_rows)
        docs_of_interest[d]["test"] = pd.DataFrame(te_rows)
        df = docs_of_interest[d]["train"]
        n_m = int(df["штатный-маркер"].sum()) if len(df) else 0
        n_act = int((df["y"] != 10000.0).sum()) if len(df) else 0
        # ослабленная калибровка: все строки дока (не только маркеры) y=10000
        all_ok = bool(len(df)) and n_act == 0
        calib_std = (n_m >= SENTINEL_MIN_ROWS
                     and bool(len(df))
                     and int((df[df["штатный-маркер"]]["y"] != 10000.0).sum()) == 0)
        docs_of_interest[d]["calib_std"] = calib_std
        docs_of_interest[d]["calib_relaxed"] = all_ok and len(df) >= SENTINEL_MIN_ROWS
        lines += [
            f"### Док {d}",
            "",
            f"- train-строк с записями дока: {len(df)} (штатных маркеров CC-без-VACV-IC: {n_m});"
            f" активных (y≠10000): {n_act}",
            f"- штатная калибровка (маркеры ≥3 и все y=10000): {calib_std}; "
            f"ослабленная (все строки дока ≥3 и все y=10000): {docs_of_interest[d]['calib_relaxed']}",
            "",
            (md_table(df) if len(df) else "_train-строк нет_"),
            "",
            f"- test-строки дока (без covered): {len(docs_of_interest[d]['test'])}",
            (md_table(docs_of_interest[d]["test"])
             if len(docs_of_interest[d]["test"]) else "_test-строк нет_"),
            "",
        ]

    # ---------- глобальный поиск CC50≈50.7 (проверка утверждения про «13»)
    hits = []
    for key, recs in records.items():
        for r in recs:
            if str(r.get("type") or "").upper() not in CC_TYPES:
                continue
            v = _to_um(r["value"], r["units"])
            if v is not None and abs(v - 50.7) <= 0.15:
                hits.append({"key": key, "type": r["type"], "value": r["value"],
                             "units": r["units"], "doc": r["doc"]})
    lines += ["## Глобальный поиск CC50≈50.7 µM по всем 853 ключам", "",
              (md_table(pd.DataFrame(hits)) if hits
               else "_Ни одной CC_TYPES-записи со значением ≈50.7 µM в ChEMBL-выгрузке нет._"),
              ""]

    # ---------- fold-internal: ослабленное правило (все строки дока → 10000)
    data = load()
    frame, y = attach_nn(attach(data.train)), np.asarray(data.y, dtype=float)
    frame = frame.reset_index(drop=True)
    frame["key"] = frame["SMILES"].map(ikey)
    art_all = pd.concat([pd.read_csv(ROOT / "data/processed/article_si.csv"),
                         pd.read_csv(ROOT / "data/processed/paper_si.csv")],
                        ignore_index=True)
    art_idx = art_all.set_index("smiles")
    key_by_smi = dict(zip(frame["SMILES"], frame["key"]))
    smi_by_key = {v: k for k, v in key_by_smi.items()}

    def relaxed_sentinels(fit_frame: pd.DataFrame, y_fit: np.ndarray,
                          base_docs: set[str]) -> set[str]:
        """Доки, калиброванные ослабленным правилом ПО ДАННОЙ выборке (fit)."""
        per_doc: dict[str, list[float]] = defaultdict(list)
        # y передаётся позиционно (fit_frame = frame.iloc[fit_mask].reset_index)
        for pos, (_, row) in enumerate(fit_frame.iterrows()):
            for r in records.get(row["key"], []):
                d = r.get("doc")
                if d and d not in base_docs:
                    per_doc[d].append(y_fit[pos])
        return {d for d, ys in per_doc.items()
                if len(ys) >= SENTINEL_MIN_ROWS and all(v == 10000.0 for v in ys)}

    splits = {f"random-{s}": [v for _, v in KFold(5, shuffle=True, random_state=s).split(y)]
              for s in (0, 1, 2, 3, 4, 5)}
    splits["scaffold"] = scaffold_folds(scaffolds_of(frame), n_splits=5)
    rows = []
    for name in ("random-0", "random-3", "scaffold"):
        src = (ROOT / "outputs/stacking-screen" if name.startswith(("random-0",))
               else ROOT / "outputs/stacking-confirm")
        probs, group = load_sources(src, name)
        pieces = fold_arrays(probs, group, frame, y, splits[name])
        for fold, piece in enumerate(pieces):
            valid = piece["valid"]
            fit_mask = np.ones(len(y), dtype=bool)
            fit_mask[valid] = False
            fit_frame = frame.iloc[fit_mask].reset_index(drop=True)
            valid_frame = frame.iloc[valid]
            sdocs_fit = calibrate_sentinel_docs(records, fit_frame, y[fit_mask])
            ptr_fit = calibrate_pair_trusted(records, fit_frame, y[fit_mask])
            relaxed_fit = relaxed_sentinels(fit_frame, y[fit_mask], set(sdocs_fit))

            lit_c = valid_lit_overrides(records, valid_frame, sdocs_fit, ptr_fit, covered)
            piece["frame"] = apply_overrides(valid_frame, art_idx, lit_c)
            mse_c = piece_mse(piece, None)

            # ослабленное применение: строка без Rule-A/покрытия, единственные
            # ChEMBL-доки — ослабленно-калиброванные, нет VACV-IC НИ В ОДНОМ доке,
            # нарушающем гипотезу (блокируем штатными _unc_vacv_ic по ДРУГИМ докам)
            new_false = new_true = 0
            for pos in range(len(valid_frame)):
                row = valid_frame.iloc[pos]
                key = row["key"]
                if row["SMILES"] in covered or candidate(records, key, pair_trusted=ptr_fit):
                    continue
                ds = {r.get("doc") for r in records.get(key, [])}
                if not ds or not ds <= relaxed_fit:
                    continue
                bd = by_doc_of(records.get(key, []))
                # гипотеза ослабленного правила: количественную VACV-IC50 ВНУТРИ
                # калиброванных доков организаторы игнорировали; но если
                # количественная активность есть в ДРУГОМ доке — не сентинель
                if any(_unc_vacv_ic(rd) for d2, rd in bd.items() if d2 not in relaxed_fit):
                    continue
                yy = float(np.asarray(piece["y"])[pos])
                if yy == 10000.0:
                    new_true += 1
                else:
                    new_false += 1
            rows.append({"split": name, "fold": fold, "mse_prod": mse_c,
                         "n_relaxed_fit": len(relaxed_fit),
                         "tgt_in_fit": {d: d in relaxed_fit for d in docs_of_interest},
                         "rel_true": new_true, "rel_false": new_false})
            print(f"{name} fold{fold}: prod {mse_c:,.0f} | relaxed-доков {len(relaxed_fit)} "
                  f"| цели в fit {rows[-1]['tgt_in_fit']} | valid верных {new_true} "
                  f"ложных {new_false}", flush=True)
    res = pd.DataFrame(rows)
    lines += ["## Fold-internal проверка ослабленного правила (fit=4/5 → valid=1/5)", "",
              md_table(res.drop(columns=["tgt_in_fit"])),
              "",
              f"- ложных сентинов на valid за {len(res)} фолдов: **{int(res['rel_false'].sum())}**;",
              f"- верных: {int(res['rel_true'].sum())}.", ""]

    # ---------- вывод
    lines += ["## Выводы", ""]
    for d, info in docs_of_interest.items():
        lines.append(f"- {d}: штатная калибровка {info['calib_std']}, "
                     f"ослабленная {info['calib_relaxed']}, "
                     f"ложных на valid {int(res['rel_false'].sum())}.")
    (OUT / "sentinel_13_45.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {OUT / 'sentinel_13_45.md'}")


if __name__ == "__main__":
    main()
