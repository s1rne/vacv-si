"""Задача 7 (диагностика no-op): верхняя граница эффекта ID50 в антивирусной половине.

Вопрос: что изменится в presence-признаках, если добавить ID50 в AMBIGUOUS_TYPES
(антивирусная сторона classify). Считаем МАКСИМАЛЬНОЕ допущение: все числовые
ID50-записи (цензурированные и нет, любые assay) засчитаны как antiviral. Если
даже при этом признаки строк не меняются — добавление доказанно no-op. Семантика
воспроизводит vacv/presence.summarise (медиана value_um по скелету, цензура по
canonical_relation) поверх records-by-key.json, как в Задаче 4; базовые
antiviral-записи = типы из AMBIGUOUS_TYPES (кроме ID50). Запрещённые колонки не
используются.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import MolWt

from vacv.classify import AMBIGUOUS_TYPES
from vacv.units import MAX_UM, MIN_UM, canonical_relation, canonical_units, to_micromolar
from vacv.match_competition import load_split
from experiments.lit_si_rule import _is_vacv

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)

BASE_TYPES = {t.replace(" ", "").upper() for t in AMBIGUOUS_TYPES}


def main() -> None:
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))
    presence = pd.read_csv(ROOT / "data/processed/competition_presence.csv", low_memory=False)
    tr, te = load_split("train"), load_split("test")
    skel: dict[str, str] = {}
    for frame in (tr, te):
        for _, row in frame.iterrows():
            if row.get("inchikey14"):
                skel.setdefault(row["inchikey14"], row["SMILES"])

    mw_by_smiles: dict[str, float | None] = {}

    def mw_of(smiles: str) -> float | None:
        if smiles not in mw_by_smiles:
            m = Chem.MolFromSmiles(smiles)
            mw_by_smiles[smiles] = MolWt(m) if m else None
        return mw_by_smiles[smiles]

    base_by_skel: dict[str, list[dict]] = {}
    id50_by_skel: dict[str, list[dict]] = {}
    id50_vacv_num = 0
    id50_dropped: Counter[str] = Counter()
    for key, recs in records.items():
        skeleton = key[:14]
        if skeleton not in skel:
            continue
        smi = skel[skeleton]
        mw = mw_of(smi)
        for r in recs:
            t = str(r.get("type") or "").strip().upper()
            if t not in BASE_TYPES and t != "ID50":
                continue
            v, reason = to_micromolar(r.get("value"), r.get("units"), mol_weight=mw)
            if v is None:
                if t == "ID50":
                    id50_dropped[str(reason)] += 1
                continue
            if not (MIN_UM <= v <= MAX_UM):
                if t == "ID50":
                    id50_dropped["outside_range"] += 1
                continue
            rel = canonical_relation(r.get("relation"))
            rec = {"value_um": v, "censored": rel not in ("=", "~"),
                   "vacv": bool(_is_vacv(r)), "type": t,
                   "assay": str(r.get("assay") or "")[:60]}
            (base_by_skel if t in BASE_TYPES else id50_by_skel).setdefault(skeleton, []).append(rec)
            if t == "ID50" and rec["vacv"]:
                id50_vacv_num += 1

    changed = []
    for skeleton, new in id50_by_skel.items():
        smi = skel[skeleton]
        rows = presence[presence["smiles"] == smi]
        if not len(rows):
            continue
        first = rows.iloc[0]
        base = base_by_skel.get(skeleton, [])
        allv = [r["value_um"] for r in base]
        newv = [r["value_um"] for r in new]
        is_train = first["split"] == "train"
        ch = []
        if first["n_antiviral"] != len(base) + len(new):
            # сверка базы с продем (куратор мог отсечь что-то); трём только дельту ID50
            ch.append(f"n_antiviral {int(first['n_antiviral'])}→{int(first['n_antiviral']) + len(new)}")
        if not bool(first["has_antiviral"]) and len(new) > 0 and len(base) == 0:
            ch.append("has_antiviral False→True")
        if bool(first["ic50_censored_any"]) != (any(r["censored"] for r in base + new)):
            ch.append("ic50_censored_any")
        if bool(first["ic50_uncensored_any"]) != (any(not r["censored"] for r in base + new)):
            ch.append("ic50_uncensored_any")
        if allv:
            med_old = float(np.median(allv))
        else:
            med_old = None if pd.isna(first["best_ic50_um"]) else float(first["best_ic50_um"])
        med_new = float(np.median(allv + newv))
        if med_old is None or abs(med_old - med_new) > 1e-9:
            ch.append(f"best_ic50_um {med_old}→{med_new:.4g}")
        if ch:
            changed.append({"split": "train" if is_train else "test",
                            "skeleton": skeleton, "smiles": smi,
                            "n_id50": len(new),
                            "id50_um": ",".join(f"{v:.3g}" for v in np.sort(newv)),
                            "id50_censored": all(r["censored"] for r in new),
                            "any_vacv_id50": any(r["vacv"] for r in new),
                            "vacv_assay": next((r["assay"] for r in new if r["vacv"]), ""),
                            "changes": "; ".join(ch)})
    ch_df = pd.DataFrame(changed)
    n_tr = int((ch_df["split"] == "train").sum()) if len(ch_df) else 0
    n_te = int((ch_df["split"] == "test").sum()) if len(ch_df) else 0

    lines = [
        "# Задача 7: верхняя граница эффекта ID50 в антивирусной половине",
        "",
        "Максимальное допущение: ВСЕ числовые ID50-записи (любые assay, цензура",
        "не важна) засчитаны как antiviral. Если изменений нет даже здесь —",
        "добавление ID50 в AMBIGUOUS_TYPES доказанно no-op для presence.",
        "",
        f"- train/test соединений с хотя бы одной конвертируемой ID50-записью: {len(id50_by_skel)};",
        f"  из них VACV-числовых записей: {id50_vacv_num};",
        f"  отброшено ID50-записей (нечисловые/вне диапазона): {dict(id50_dropped) or 'нет'};",
        f"- строк (train/test) с изменением хотя бы одного presence-признака: **{n_tr} / {n_te}**.",
        "",
    ]
    if len(ch_df):
        head = "| " + " | ".join(map(str, ch_df.columns)) + " |"
        sep = "|" + "|".join("---" for _ in ch_df.columns) + "|"
        body = "\n".join("| " + " | ".join(map(str, r)) + " |"
                         for r in ch_df.itertuples(index=False))
        lines += [head, sep, body, ""]
    else:
        lines += ["_Ни одна строка не меняется даже при максимальном допущении._", ""]
    (OUT / "id50_presence_effect.md").write_text("\n".join(lines), encoding="utf-8")
    if len(ch_df):
        ch_df.to_csv(OUT / "id50_presence_effect.csv", index=False)
    print(f"скелетов с числовыми ID50: {len(id50_by_skel)}; VACV-числовых записей: {id50_vacv_num}; "
          f"отброшено: {dict(id50_dropped) or 'нет'}")
    print(f"изменённых строк: train {n_tr}, test {n_te}")
    if len(ch_df):
        print(ch_df.to_string(index=False))
    print(f"→ {OUT / 'id50_presence_effect.md'}")


if __name__ == "__main__":
    main()
