"""Батч-сбор activity-записей, ключ — полный InChIKey строки.

Старый chembl_batch_records.py ключевал по Title, а Title повторяются между
train/test с РАЗНЫМИ SMILES (например, '9a' — камфора-амид в train и
имидазопиридин в test) — записи перезаписывались первым вхождением.
Здесь ключ = InChIKey каждой уникальной SMILES.

Переиспользует title-molecules.json (Title → молекулы первого вхождения).
Выход: outputs/doc-scraper/records-by-key.json {inchikey: [records]}
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
    old_map = json.load(open(OUT / "title-molecules.json"))  # Title → молекулы 1-го вхождения

    # inchikey каждой строки
    keys: dict[str, str] = {}
    for row in lookup.itertuples():
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            continue
        k = MolToInchiKey(mol)
        if k:
            keys[row.smiles] = k

    # переиспользование: Title → (первый SMILES) → inchikey → молекулы
    first_smiles: dict[str, str] = {}
    for row in lookup.itertuples():
        first_smiles.setdefault(str(row.Title), row.smiles)

    key_mols: dict[str, list[str]] = {}
    for title, mols in old_map.items():
        smi = first_smiles.get(title)
        if not smi:
            continue
        k = keys.get(smi)
        if k:
            key_mols[k] = mols

    todo = [k for smi, k in keys.items() if k not in key_mols]
    print(f"уникальных SMILES: {len(keys)}, переиспользовано: {len(key_mols)}, "
          f"к резолвингу: {len(todo)}", flush=True)
    for i, k in enumerate(todo, 1):
        key_mols[k] = molecules_for(k)
        if i % 100 == 0:
            json.dump(key_mols, open(OUT / "key-molecules.json", "w"))
            print(f"molecules {i}/{len(todo)}", flush=True)
        time.sleep(0.05)
    json.dump(key_mols, open(OUT / "key-molecules.json", "w"))

    unique_ids = sorted({m for ids in key_mols.values() for m in ids})
    print(f"{len(key_mols)} inchikey → {len(unique_ids)} unique molecules", flush=True)

    records_path = OUT / "records-by-key.json"
    cache = json.load(open(records_path)) if records_path.exists() else {}
    done_ids = {r["molecule_chembl_id"] for recs in cache.values() for r in recs}
    todo_ids = [m for m in unique_ids if m not in done_ids]
    print(f"fetching activities for {len(todo_ids)} molecules", flush=True)
    fetched = batched_activities(todo_ids)
    for k, ids in key_mols.items():
        if k in cache:
            continue
        cache[k] = [r for mid in ids for r in fetched.get(mid, [])]
    json.dump(cache, open(records_path, "w"))
    n_with = sum(1 for v in cache.values() if v)
    print(f"done: {len(cache)} inchikey, {n_with} с записями")


if __name__ == "__main__":
    main()
