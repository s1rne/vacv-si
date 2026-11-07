"""Задача 7: является ли ID50 старым названием IC50 (расширение антивирусного матчера).

Вопрос по заданиям: 46 VACV-записей ID50 — если там, где у соединения есть и ID50, и
IC50, значения бит-в-бит совпадают, то ID50 — синоним IC50 и его можно добавить в
антивирусную сторону классификатора (`AMBIGUOUS_TYPES` в vacv/classify.py; в
парном матчере `experiments/lit_si_rule.py` IC_TYPES уже содержит ID50).

Сверка y-НЕЗАВИСИМАЯ: сравниваются два ChEMBL-поля записей между собой
(ID50 vs IC50 по ключу/доку/assay) — утечки target-информации нет. Запрещённые
колонки (IC50_uM_VACV) — только локальная диагностика точности.

Выход: outputs/qwen-sentinel/id50_alias.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

from vacv.classify import AMBIGUOUS_TYPES
from experiments.lit_si_rule import IC_TYPES, _is_vacv, _to_um

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)

TOL_EXACT = 0.003


def ikey(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return MolToInchiKey(m) if m else ""


def main() -> None:
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    train["key"] = train["SMILES"].map(ikey)
    test["key"] = test["SMILES"].map(ikey)
    comp_keys = set(train["key"]) | set(test["key"])
    key2smi: dict[str, str] = {}
    for f in (train, test):
        for _, r in f.iterrows():
            key2smi.setdefault(r["key"], r["SMILES"])

    # --- сбор записей
    n_id50 = n_id50_vacv = 0
    pairs_same_doc = []          # (ID50-запись, ближайшая IC50 того же дока/ключа)
    keys_id50 = set()
    keys_both = set()
    vacv_expand_keys = set()     # есть несценз. VACV ID50, нет несценз. VACV IC50
    per_key: dict[str, dict] = {}
    for key, recs in records.items():
        id50 = [r for r in recs if str(r.get("type") or "").upper() == "ID50"]
        ic50 = [r for r in recs if str(r.get("type") or "").upper() == "IC50"]
        if not id50:
            continue
        keys_id50.add(key)
        n_id50 += len(id50)
        n_id50_vacv += sum(1 for r in id50 if _is_vacv(r))
        unc_id50_vacv = [r for r in id50 if _is_vacv(r)
                         and (r.get("relation") or "").strip() not in (">", ">=")
                         and _to_um(r["value"], r["units"]) is not None]
        unc_ic50_vacv = [r for r in ic50 if _is_vacv(r)
                         and (r.get("relation") or "").strip() not in (">", ">=")
                         and _to_um(r["value"], r["units"]) is not None]
        if unc_id50_vacv and not unc_ic50_vacv and not any(
                str(r.get("type") or "").upper() in AMBIGUOUS_TYPES and _is_vacv(r)
                and (r.get("relation") or "").strip() not in (">", ">=")
                and _to_um(r["value"], r["units"]) is not None for r in recs):
            vacv_expand_keys.add(key)
        if id50 and ic50:
            keys_both.add(key)
        for r in id50:
            v = _to_um(r["value"], r["units"])
            if v is None:
                continue
            cands = [(_to_um(c["value"], c["units"]), c) for c in ic50]
            cands = [(cv, c) for cv, c in cands if cv is not None]
            if not cands:
                continue
            # ближайшее по лог-шкале
            cv, c = min(cands, key=lambda t: abs(np.log(t[0] / v)))
            ratio = v / cv
            pairs_same_doc.append({
                "key": key, "in_comp": key in comp_keys,
                "same_doc": c.get("doc") == r.get("doc"),
                "vacv": _is_vacv(r) and _is_vacv(c),
                "id50_um": round(v, 4), "ic50_um": round(cv, 4),
                "ratio": round(ratio, 4),
                "exact": abs(ratio - 1) <= TOL_EXACT,
                "near2x": 0.5 <= ratio <= 2,
                "relation_id50": str(r.get("relation") or ""),
            })
        per_key[key] = {"id50": len(id50), "ic50": len(ic50)}

    pairs = pd.DataFrame(pairs_same_doc)
    n_cmp = len(pairs)
    exact = int(pairs["exact"].sum()) if n_cmp else 0
    near = int(pairs["near2x"].sum()) if n_cmp else 0
    same_doc_df = pairs[pairs["same_doc"]] if n_cmp else pairs
    vacv_df = pairs[pairs["vacv"]] if n_cmp else pairs

    # --- локальная диагностика по запрещённой колонке (только оценка)
    tr_hits = []
    for _, row in train.iterrows():
        ic_true = row.get("IC50_uM_VACV")
        if pd.isna(ic_true):
            continue
        id50v = [_to_um(r["value"], r["units"]) for r in records.get(row["key"], [])
                 if str(r.get("type") or "").upper() == "ID50" and _is_vacv(r)
                 and _to_um(r["value"], r["units"]) is not None]
        if not id50v:
            continue
        med = float(np.median(id50v))
        tr_hits.append({"Title": str(row["Title"]), "id50_um": round(med, 4),
                        "IC50_uM_VACV": float(ic_true),
                        "ratio": round(med / float(ic_true), 4),
                        "exact": abs(med / float(ic_true) - 1) <= TOL_EXACT})
    hits = pd.DataFrame(tr_hits)

    # --- покрытие: какие train/test строки получили бы антивирусное покрытие
    def rows_on(keys: set[str], frame: pd.DataFrame) -> int:
        return int(frame["key"].isin(keys).sum())

    expand_train = rows_on(vacv_expand_keys, train)
    expand_test = rows_on(vacv_expand_keys, test)

    def md_table(df: pd.DataFrame, cols=None) -> str:
        d = df if cols is None else df[cols]
        if d is None or not len(d):
            return "_нет_"
        head = "| " + " | ".join(map(str, d.columns)) + " |"
        sep = "|" + "|".join("---" for _ in d.columns) + "|"
        body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False))
        return "\n".join([head, sep, body])

    lines = [
        "# Задача 7: ID50 как синоним IC50",
        "",
        "Сверка y-независимая (запись-vs-запись); запрещённые колонки — только",
        "локальная диагностика. Контекст: `AMBIGUOUS_TYPES` в `vacv/classify.py` =",
        f"{sorted(AMBIGUOUS_TYPES)} — ID50 **отсутствует** (→ unknown → выбрасывается);",
        "в парном матчере `experiments/lit_si_rule.py` `IC_TYPES` **уже содержит ID50**,",
        "так что для него менять нечего — вопрос только про антивирусную половину",
        "presence/classify.",
        "",
        "## Итоги сверки",
        "",
        f"- ID50-записей: {n_id50} (VACV: {n_id50_vacv}); ключей с ID50: {len(keys_id50)};",
        f"  ключей с обоими типами (ID50+IC50): {len(keys_both)}.",
        f"- Сравниваемых пар (ID50 ↔ ближайшая IC50 того же ключа): {n_cmp};",
        f"  бит-в-бит (≤0.3%): **{exact}** ({exact / n_cmp:.0%} если n_cmp);",
        f"  в пределах ×2: {near} ({near / n_cmp:.0%} если n_cmp).",
        f"- Подвыборка тот же документ: {len(same_doc_df)} пар, бит-в-бит "
        f"{int(same_doc_df['exact'].sum()) if len(same_doc_df) else 0}.",
        f"- Подвыборка VACV-обоюдные: {len(vacv_df)} пар, бит-в-бит "
        f"{int(vacv_df['exact'].sum()) if len(vacv_df) else 0}.",
        "",
        "## Расширение покрытия (если добавлять ID50 в антивирусную половину)",
        "",
        f"- Ключей с несцензурированной VACV ID50, но без какой-либо несцензурированной",
        f"  VACV-записи типов-потентности: {len(vacv_expand_keys)} → затронуло бы",
        f"  {expand_train} train- и {expand_test} test-строк.",
        "",
        "## Локальная диагностика против запрещённой IC50_uM_VACV (только оценка)",
        "",
        (f"Строк train с VACV ID50-записью и известной правдой: {len(hits)}; бит-в-бит: "
         f"{int(hits['exact'].sum()) if len(hits) else 0}." if len(hits)
         else "Нет строк train с VACV-записями ID50 и известной IC50_uM_VACV."),
        "",
        md_table(hits) if len(hits) else "",
        "",
        "## Детали пар (train/test соединения, выборка)",
        "",
        md_table(pairs[pairs["in_comp"]].sort_values("ratio", key=lambda s: (s - 1).abs())
                 .drop_duplicates("key").head(30)) if n_cmp else "_нет_",
        "",
        "## Выводы",
        "",
        "см. финализацию (автозаполняется числами выше)",
    ]
    (OUT / "id50_alias.md").write_text("\n".join(lines), encoding="utf-8")
    pairs.to_csv(OUT / "id50_alias_pairs.csv", index=False)
    print(f"ID50: {n_id50} записей, VACV {n_id50_vacv}; keys id50={len(keys_id50)}, both={len(keys_both)}")
    print(f"пар сравнения {n_cmp}: бит-в-бит {exact}, x2 {near}; same-doc {len(same_doc_df)} "
          f"(бит {int(same_doc_df['exact'].sum()) if len(same_doc_df) else 0}); "
          f"vacv {len(vacv_df)} (бит {int(vacv_df['exact'].sum()) if len(vacv_df) else 0})")
    print(f"расширение: keys {len(vacv_expand_keys)}, train {expand_train}, test {expand_test}")
    print(f"диагностика vs IC50_uM_VACV: {len(hits)} строк, бит-в-бит {int(hits['exact'].sum()) if len(hits) else 0}")
    if len(hits):
        print(hits.to_string(index=False))


if __name__ == "__main__":
    main()
