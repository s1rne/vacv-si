"""Связь сентинела с relation записей ChEMBL для VACV-потенций.

Гипотеза: колонка IC50_uM_VACV соревнования берётся из конкретной записи ChEMBL
(документ × молекула × assay). Если у записи relation '>' (цензурированный
EC50/IC50 — «не активен до 100 µM»), соревнование пишет IC50 = NaN и
SI = 10000 (сентинел). Проверка на train: достаём по полному InChIKey все
activity-записи молекулы, фильтруем до VACV-потаенций (упоминание vaccinia
в assay), смотрим relation.

Выход: outputs/doc-scraper/train-relations.json
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

BASE = "https://www.ebi.ac.uk/chembl/api/data"
OUT = Path("outputs/doc-scraper")
POTENT_TYPES = {"IC50", "EC50", "ED50", "ID50", "CC50", "TD50", "MCC", "AC50"}
VACCINIA_RE = re.compile(r"vaccinia|vacv|cowpox", re.I)


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def molecule_ids(inchikey: str) -> list[str]:
    """Молекулы по полному InChIKey; соли/стереоизомеры — по 14-символьному префиксу."""
    d = fetch(f"{BASE}/molecule.json?molecule_structures__standard_inchi_key={inchikey}&limit=10")
    ids = [m["molecule_chembl_id"] for m in d.get("molecules", [])]
    if not ids:
        d = fetch(f"{BASE}/molecule.json?molecule_structures__standard_inchi_key__startswith="
                  f"{inchikey[:14]}&limit=25")
        ids = [m["molecule_chembl_id"] for m in d.get("molecules", [])]
    return ids


def _to_um(value, units) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    u = str(units or "")
    if u == "nM":
        return v / 1000.0
    if u in ("µM", "μM", "uM", "M"):
        return v if u != "M" else v * 1e6
    return None  # µg/mL и прочие требуют молекулярной массы — пропускаем


def all_activities(mol: str) -> list[dict]:
    page = fetch(f"{BASE}/activity.json?molecule_chembl_id={mol}&limit=1000")
    recs = page["activities"]
    while page["page_meta"].get("next"):
        nxt = page["page_meta"]["next"]
        if nxt.startswith("/"):
            nxt = "https://www.ebi.ac.uk" + nxt
        page = fetch(nxt)
        recs.extend(page["activities"])
    time.sleep(0.1)
    return recs


def potency_records(recs: list[dict]) -> list[dict]:
    out = []
    for r in recs:
        if r.get("standard_type") not in POTENT_TYPES:
            continue
        try:
            value = float(r.get("standard_value"))
        except (TypeError, ValueError):
            continue
        text = " ".join(str(r.get(k) or "") for k in
                        ("assay_description", "assay_type", "target_pref_name", "document_journal"))
        if not VACCINIA_RE.search(text):
            continue
        out.append({
            "doc": r.get("document_chembl_id"),
            "type": r.get("standard_type"),
            "value": value,
            "units": r.get("standard_units"),
            "relation": r.get("standard_relation"),
            "assay": str(r.get("assay_description"))[:120],
            "year": r.get("document_year"),
        })
    return out


def main() -> None:
    lookup = pd.read_csv("data/processed/competition_open_data_lookup.csv", low_memory=False)
    train = pd.read_csv("data/raw/kaggle/train.csv")
    test = pd.read_csv("data/raw/kaggle/test.csv")

    cache_path = OUT / "molecule-relations.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}

    rows = []
    for split, frame in (("train", train), ("test", test)):
        for record in frame.to_dict("records"):
            title = str(record["Title"])
            if title in cache:
                recs = cache[title]
            else:
                mol = Chem.MolFromSmiles(record["SMILES"])
                key = MolToInchiKey(mol) if mol else ""
                recs = []
                if key:
                    for mid in molecule_ids(key):
                        recs.extend(potency_records(all_activities(mid)))
                cache[title] = recs
                json.dump(cache, open(cache_path, "w"))
                time.sleep(0.1)
            ic50 = [r for r in recs if r["type"] in ("IC50", "EC50", "ED50", "ID50")]
            cc = [r for r in recs if r["type"] in ("CC50", "TD50", "MCC")]
            rows.append({
                "split": split, "Title": title,
                "n_ic50": len(ic50), "n_cc50": len(cc),
                "ic50_relations": sorted({r["relation"] or "" for r in ic50}),
                "ic50_values_um": sorted({round(v, 4) for v in
                                          (_to_um(r["value"], r["units"]) for r in ic50) if v}),
                "cc50_relations": sorted({r["relation"] or "" for r in cc}),
                "cc50_values_um": sorted({round(v, 4) for v in
                                          (_to_um(r["value"], r["units"]) for r in cc) if v}),
            })
        print(f"{split}: {len(rows)} rows cached", flush=True)

    df = pd.DataFrame(rows)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    y = train["SI VACV"].to_numpy()
    sentinel = (y == 10000) & train["IC50_uM_VACV"].isna().to_numpy()

    # правило: есть IC50-записи, и все с rel '>' → сентинел
    rule = tr["n_ic50"].gt(0) & tr["ic50_relations"].apply(lambda rs: set(rs) == {">"})
    tp = int((rule & sentinel).sum()); fp = int((rule & ~sentinel).sum())
    fn = int((~rule & sentinel).sum())
    print(f"\nrule(all IC50 rel='>') on train: TP={tp} FP={fp} FN={fn} "
          f"precision={tp / max(tp + fp, 1):.3f} recall={tp / max(tp + fn, 1):.3f}")
    fp_rows = tr[rule & ~sentinel]
    if len(fp_rows):
        print("FP:")
        for _, r in fp_rows.iterrows():
            yi = y[list(tr["Title"]).index(r["Title"])]
            print(f"  {r['Title'][:20]:20} y={yi} ic50rels={r['ic50_relations']} vals={r['ic50_values_um']}")
    fn_rows = tr[sentinel & ~rule]
    print(f"FN: {len(fn_rows)} (сентинелы без '>' — записи '=' или их нет)")

    # обратное: строки с '=' записью → не сентинел?
    has_eq = tr["ic50_relations"].apply(lambda rs: "=" in rs or "~" in rs)
    print(f"rows with '=' IC50 record: {has_eq.sum()}, of them sentinels: "
          f"{int((has_eq & sentinel).sum())}")

    json.dump(cache, open(cache_path, "w"))
    df.to_json(OUT / "train-relations.json", orient="records", indent=1)


if __name__ == "__main__":
    main()
