"""Задача 7: патч presence для проверки «ID50 в антивирусную половину».

Строит outputs/qwen-sentinel/presence_id50av.csv — копию
data/processed/competition_presence.csv, где к антивирусной стороне добавлены
ID50-записи ТАК, как их увидел бы реальный конвейер при ID50 ∈ AMBIGUOUS_TYPES:
  classify(standard_type="ID50", description=assay) == "antiviral"
  И mentions_vaccinia(assay) (фильтр antiviral_side)
  И числовое значение в MIN_UM..MAX_UM (curate).
Запись о biochemical-формате в records-by-key.json отсутствует — единственное
несоответствие продем, задокументировано (верхняя оценка).
Для строк с n_antiviral==0 признаки задаются целиком новыми записями; для
строк с уже имеющимися antiviral-записями — n_antiviral += новых, флаги
цензуры |=, best_ic50_um не меняется (сырых базовых значений локально нет —
консервативно). Ничего не трогает, еслиfaithful-записей 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import MolWt

import vacv.classify as classify_mod
from vacv.classify import AMBIGUOUS_TYPES, classify, mentions_vaccinia
from vacv.units import MAX_UM, MIN_UM, canonical_relation, to_micromolar
from vacv.match_competition import load_split

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

    # ID50 как антивирусная запись — через НАСТОЯЩИЙ classify с расширенным набором
    patched_set = set(classify_mod.AMBIGUOUS_TYPES) | {"ID50"}
    saved = classify_mod.AMBIGUOUS_TYPES
    classify_mod.AMBIGUOUS_TYPES = patched_set
    try:
        by_skel: dict[str, list[dict]] = {}
        cls_counter: dict[str, int] = {}
        for key, recs in records.items():
            skeleton = key[:14]
            if skeleton not in skel:
                continue
            smi = skel[skeleton]
            mw = mw_of(smi)
            for r in recs:
                if str(r.get("type") or "").strip().upper() != "ID50":
                    continue
                assay = str(r.get("assay") or "")
                cls = classify("ID50", description=assay or None)
                cls_counter[cls] = cls_counter.get(cls, 0) + 1
                if cls != "antiviral" or not mentions_vaccinia(assay or None):
                    continue
                v, _ = to_micromolar(r.get("value"), r.get("units"), mol_weight=mw)
                if v is None or not (MIN_UM <= v <= MAX_UM):
                    continue
                rel = canonical_relation(r.get("relation"))
                by_skel.setdefault(skeleton, []).append(
                    {"value_um": v, "censored": rel not in ("=", "~")})
    finally:
        classify_mod.AMBIGUOUS_TYPES = saved

    patched = presence.copy()
    n_fill, n_delta = 0, 0
    detail = []
    for skeleton, new in by_skel.items():
        smi = skel[skeleton]
        rows = patched[patched["smiles"] == smi]
        if not len(rows):
            continue
        vals = np.array([r["value_um"] for r in new])
        for i in rows.index:
            first = patched.loc[i]
            if int(first["n_antiviral"]) == 0:
                patched.loc[i, "n_antiviral"] = len(new)
                patched.loc[i, "has_antiviral"] = True
                patched.loc[i, "best_ic50_um"] = float(np.median(vals))
                patched.loc[i, "ic50_censored_any"] = any(r["censored"] for r in new)
                patched.loc[i, "ic50_uncensored_any"] = any(not r["censored"] for r in new)
                n_fill += 1
                detail.append({"split": first["split"], "smiles": smi, "n_new": len(new),
                               "best_new_um": round(float(np.median(vals)), 4),
                               "mode": "fill"})
            else:
                patched.loc[i, "n_antiviral"] = int(first["n_antiviral"]) + len(new)
                patched.loc[i, "ic50_censored_any"] = bool(first["ic50_censored_any"]) or any(
                    r["censored"] for r in new)
                patched.loc[i, "ic50_uncensored_any"] = bool(first["ic50_uncensored_any"]) or any(
                    not r["censored"] for r in new)
                n_delta += 1
                detail.append({"split": first["split"], "smiles": smi, "n_new": len(new),
                               "best_new_um": None, "mode": "count-only"})
    patched.to_csv(OUT / "presence_id50av.csv", index=False)
    df = pd.DataFrame(detail)
    n_tr = int((df["split"] == "train").sum()) if len(df) else 0
    n_te = int((df["split"] == "test").sum()) if len(df) else 0
    (OUT / "id50_av_patch.md").write_text(
        "\n".join([
            "# Задача 7: патч presence «ID50 → антивирусная половина»",
            "",
            f"Классификация честным `classify()` при ID50 ∈ AMBIGUOUS_TYPES (assay=текст записи):",
            f"распределение классов ID50-записей: {cls_counter}.",
            f"Скелетов с faithful-antiviral ID50-записями: {len(by_skel)}.",
            f"Строк изменено: train {n_tr}, test {n_te} (fill {n_fill}, count-only {n_delta}).",
            "Для fill-строк все антивирусные признаки задаются новыми записями;",
            "для count-only — только счётчик/флаги цензуры (best_ic50_um консервативно",
            "не меняется: сырых базовых значений локально нет). is_biochemical не",
            "воспроизводится (нет assay_format) — верхняя оценка.",
            "",
            "| split | smiles | n_new | best_new_um | mode |",
            "|---|---|---|---|---|",
            *[f"| {r['split']} | `{r['smiles'][:44]}` | {r['n_new']} | {r['best_new_um']} | {r['mode']} |"
              for r in detail],
        ]),
        encoding="utf-8")
    print(f"classify ID50: {cls_counter}")
    print(f"faithful-скелетов: {len(by_skel)}; строк: train {n_tr}, test {n_te} "
          f"(fill {n_fill}, count-only {n_delta})")
    print(f"→ {OUT / 'presence_id50av.csv'}, {OUT / 'id50_av_patch.md'}")


if __name__ == "__main__":
    main()
