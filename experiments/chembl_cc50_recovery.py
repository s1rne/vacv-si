"""Восстановление CC50-записей для всех соединений соревнования.

CYTOTOX_TYPES в classify.py не содержит 'MCC' (minimum cytotoxic concentration) —
записи цитотоксичности из статей с MCC-колонкой выпадали из cytotox-стороны,
и соединения ошибочно становились ic50-only. Соревнование же эти записи
использует (train 24a11: CC50_uM=100.0 — это MCC «>100» → 100).

Проход по всем 917 молекулам: все activity-записи типов
CC50/TD50/MCC/TC50/MTC/CT50/LC50/MNTD/CD50/GI50/TGI + IC50/EC50 с relation.
Выход: outputs/doc-scraper/cytotox-records.json
  {Title: {"cc50": [{doc, type, value_um, relation, assay}],
           "ic50": [...]}}
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

BASE = "https://www.ebi.ac.uk/chembl/api/data"
OUT = Path("outputs/doc-scraper")

CC_TYPES = {"CC50", "TD50", "MCC", "TC50", "MTC", "CT50", "LC50", "MNTD", "CD50", "GI50", "TGI", "MTD"}
IC_TYPES = {"IC50", "EC50", "ED50", "ID50", "AC50"}


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def molecule_ids(inchikey: str) -> list[str]:
    d = fetch(f"{BASE}/molecule.json?molecule_structures__standard_inchi_key={inchikey}&limit=10")
    ids = [m["molecule_chembl_id"] for m in d.get("molecules", [])]
    if not ids:
        d = fetch(f"{BASE}/molecule.json?molecule_structures__standard_inchi_key__startswith="
                  f"{inchikey[:14]}&limit=25")
        ids = [m["molecule_chembl_id"] for m in d.get("molecules", [])]
    return ids


def all_activities(mol: str) -> list[dict]:
    page = fetch(f"{BASE}/activity.json?molecule_chembl_id={mol}&limit=1000")
    recs = page["activities"]
    while page["page_meta"].get("next"):
        nxt = page["page_meta"]["next"]
        if nxt.startswith("/"):
            nxt = "https://www.ebi.ac.uk" + nxt
        page = fetch(nxt)
        recs.extend(page["activities"])
    return recs


def to_um(value, units):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    u = str(units or "")
    if u == "nM":
        return v / 1000.0
    if u in ("µM", "μM", "uM"):
        return v
    if u == "mM":
        return v * 1e3
    return None


def main() -> None:
    cache_path = OUT / "cytotox-records.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    lookup = pd.read_csv("data/processed/competition_open_data_lookup.csv", low_memory=False)
    lookup = lookup.drop_duplicates("Title")

    for i, row in enumerate(lookup.itertuples(), 1):
        title = str(row.Title)
        if title in cache:
            continue
        mol = Chem.MolFromSmiles(row.smiles)
        key = MolToInchiKey(mol) if mol else ""
        cc, ic = [], []
        if key:
            for mid in molecule_ids(key):
                for r in all_activities(mid):
                    rtype = r.get("standard_type")
                    v = to_um(r.get("standard_value"), r.get("standard_units"))
                    if v is None:
                        continue
                    rec = {"doc": r.get("document_chembl_id"), "type": rtype, "value_um": v,
                           "relation": r.get("standard_relation") or "",
                           "assay": str(r.get("assay_description"))[:110]}
                    if rtype in CC_TYPES:
                        cc.append(rec)
                    elif rtype in IC_TYPES:
                        ic.append(rec)
        cache[title] = {"cc50": cc, "ic50": ic}
        if i % 25 == 0:
            json.dump(cache, open(cache_path, "w"))
            print(f"{i}/{len(lookup)} cached", flush=True)
        time.sleep(0.05)
    json.dump(cache, open(cache_path, "w"))
    n_cc = sum(1 for v in cache.values() if v["cc50"])
    print(f"done: {len(cache)} compounds, {n_cc} with cytotox records")


if __name__ == "__main__":
    main()
