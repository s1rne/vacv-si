"""Записи ChEMBL по документам-источникам ic50-only соединений.

Для 77 ic50-only SMILES (IC50 известен, CC50 нет) известны документы ChEMBL,
из которых взят IC50. Здесь запрашиваются ВСЕ activity-записи этих документов:
если в той же статье CC50 курирован, его можно восстановить без скрейпинга
таблиц — сопоставив молекулу по InChIKey14, а тип записи по standard_type.

Выход: outputs/doc-scraper/doc-activities.json
  {doc_id: {"records": [{activity_id, molecule_chembl_id, type, value, units,
                         relation, assay_chembl_id, assay_desc, comment,
                         smiles, inchikey14}]}}
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

BASE = "https://www.ebi.ac.uk/chembl/api/data"
OUT = Path("outputs/doc-scraper")


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def activities_of(doc: str) -> list[dict]:
    url = f"{BASE}/activity.json?document_chembl_id={doc}&limit=1000"
    out = []
    while url:
        page = get_json(url)
        out.extend(page["activities"])
        url = page["page_meta"].get("next")
        if url and url.startswith("/"):
            url = "https://www.ebi.ac.uk" + url
        time.sleep(0.1)
    return out


def inchikey14(smiles: str | None) -> str:
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    key = MolToInchiKey(mol)
    return key[:14] if key else ""


def main() -> None:
    mapping = json.load(open("outputs/ic50_only_docs.json"))
    doc_ids = sorted({d for v in mapping.values() for d in v})
    cache_path = OUT / "doc-activities.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    for i, doc in enumerate(doc_ids):
        if doc in cache:
            continue
        records = activities_of(doc)
        cache[doc] = {
            "records": [
                {
                    "activity_id": r.get("activity_id"),
                    "molecule_chembl_id": r.get("molecule_chembl_id"),
                    "type": r.get("standard_type"),
                    "value": r.get("standard_value"),
                    "units": r.get("standard_units"),
                    "relation": r.get("standard_relation"),
                    "assay_chembl_id": r.get("assay_chembl_id"),
                    "assay_desc": r.get("assay_description"),
                    "comment": r.get("activity_comment"),
                    "smiles": r.get("canonical_smiles"),
                    "inchikey14": inchikey14(r.get("canonical_smiles")),
                }
                for r in records
            ]
        }
        types = sorted({str(r.get("standard_type")) for r in records})
        print(f"{i + 1}/{len(doc_ids)} {doc}: {len(records)} records, types={types}", flush=True)
        json.dump(cache, open(cache_path, "w"))
        time.sleep(0.2)
    print("done:", len(cache), "documents")


if __name__ == "__main__":
    main()
