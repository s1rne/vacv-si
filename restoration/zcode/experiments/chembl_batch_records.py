"""Батч-сбор всех activity-записей для всех соединений соревнования.

molecule_chembl_id__in позволяет тянуть по 50 молекул за запрос — на два
порядка быстрее поштучных вызовов. Собираем ВСЕ записи с числовым
standard_value; фильтрация (vaccinia / cytotox-типы) делается локально.

Выход: outputs/doc-scraper/all-activity-records.json
  {Title: [{"molecule_chembl_id", "doc", "type", "value", "units",
            "relation", "assay"}]}
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
BATCH = 50


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def molecules_for(inchikey: str) -> list[str]:
    d = fetch(f"{BASE}/molecule.json?molecule_structures__standard_inchi_key={inchikey}&limit=10")
    ids = [m["molecule_chembl_id"] for m in d.get("molecules", [])]
    if not ids:
        d = fetch(f"{BASE}/molecule.json?molecule_structures__standard_inchi_key__startswith="
                  f"{inchikey[:14]}&limit=25")
        ids = [m["molecule_chembl_id"] for m in d.get("molecules", [])]
    return ids


def batched_activities(mol_ids: list[str]) -> dict[str, list[dict]]:
    """Все записи для списка молекул, сгруппированные по molecule_chembl_id."""
    out: dict[str, list[dict]] = {}
    for i in range(0, len(mol_ids), BATCH):
        chunk = mol_ids[i:i + BATCH]
        url = f"{BASE}/activity.json?molecule_chembl_id__in={','.join(chunk)}&limit=1000"
        while url:
            page = fetch(url)
            for r in page["activities"]:
                try:
                    float(r.get("standard_value"))
                except (TypeError, ValueError):
                    continue
                out.setdefault(r["molecule_chembl_id"], []).append({
                    "doc": r.get("document_chembl_id"),
                    "type": r.get("standard_type"),
                    "value": r.get("standard_value"),
                    "units": r.get("standard_units"),
                    "relation": r.get("standard_relation"),
                    "assay": str(r.get("assay_description"))[:110],
                })
            nxt = page["page_meta"].get("next")
            url = ("https://www.ebi.ac.uk" + nxt) if nxt and nxt.startswith("/") else nxt
        time.sleep(0.2)
    return out


def main() -> None:
    lookup = pd.read_csv("data/processed/competition_open_data_lookup.csv", low_memory=False)
    lookup = lookup.drop_duplicates("Title").reset_index(drop=True)

    mol_path = OUT / "title-molecules.json"
    mol_map = json.load(open(mol_path)) if mol_path.exists() else {}
    for i, row in enumerate(lookup.itertuples(), 1):
        title = str(row.Title)
        if title in mol_map:
            continue
        mol = Chem.MolFromSmiles(row.smiles)
        key = MolToInchiKey(mol) if mol else ""
        mol_map[title] = molecules_for(key) if key else []
        if i % 100 == 0:
            json.dump(mol_map, open(mol_path, "w"))
            print(f"molecules {i}/{len(lookup)}", flush=True)
        time.sleep(0.05)
    json.dump(mol_map, open(mol_path, "w"))
    unique_ids = sorted({m for ids in mol_map.values() for m in ids})
    print(f"{len(mol_map)} titles → {len(unique_ids)} unique molecules", flush=True)

    records_path = OUT / "all-activity-records.json"
    cache = json.load(open(records_path)) if records_path.exists() else {}
    done_ids = {r["molecule_chembl_id"] for recs in cache.values() for r in recs}
    todo_ids = [m for m in unique_ids if m not in done_ids]
    print(f"fetching activities for {len(todo_ids)} molecules", flush=True)
    fetched = batched_activities(todo_ids)
    for title, ids in mol_map.items():
        if title in cache:
            continue
        cache[title] = [r for mid in ids for r in fetched.get(mid, [])]
    json.dump(cache, open(records_path, "w"))
    print(f"done: {len(cache)} titles")


if __name__ == "__main__":
    main()
