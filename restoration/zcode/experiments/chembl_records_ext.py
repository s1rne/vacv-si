"""Дозапись records-by-key: полные описания assay + молекулы через PubChem SID.

Скребок chembl_records_by_key.py режет assay_description до 110 символов —
в VR-скрине JMC 1984 (док CHEMBL1122625) хвост "(T) и Vaccinia virus control
cells" терялся, записи Virus rating были невидимы для vaccinia-фильтра майнера.
Здесь для ключей с записями в целевых доках:
  1. molecules_for(key) (точный + startswith-поиск) и SID_MOLS (PubChem
     SID → ChEMBL molecule, для ключей с пустым key-search);
  2. activities с ПОЛНЫМИ описаниями (обрезка 400);
  3. слияние ТОЛЬКО записей целевых доков: сигнатура
     (doc, type, value, units, relation) — апгрейд assay, иначе append.

Целевые доки: CHEMBL1122625 (VR/Toxic, JMC 1984), CHEMBL1127375 (MCC/MIC
панель E6SM), CHEMBL1147827 (EC50 CPE vs plaque, HFF), CHEMBL1926544
(JMC 2011, HFF-панель).

Запуск: uv run python experiments/chembl_records_ext.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

BASE = "https://www.ebi.ac.uk/chembl/api/data"
OUT = Path("outputs/doc-scraper/records-by-key.json")
BATCH = 50

#: Доки, записи которых дополняются/освежаются
TARGET_DOCS = {"CHEMBL1122625", "CHEMBL1127375", "CHEMBL1147827", "CHEMBL1926544"}

#: InChIKey → molecule_chembl_id через PubChem substance (key-search пуст)
SID_MOLS = {
    # sid103202079 (8-азагуанин, Vero): deposit-структура вне key-search
    "OANWYRGSXWEMDQ-UHFFFAOYSA-N": "CHEMBL31194",
}


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


def activities_for(mol_ids: list[str]) -> list[dict]:
    """Полные activity-записи (описание до 400 символов) молекул."""
    out = []
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
                out.append({
                    "doc": r.get("document_chembl_id"),
                    "type": r.get("standard_type"),
                    "value": r.get("standard_value"),
                    "units": r.get("standard_units"),
                    "relation": r.get("standard_relation"),
                    "assay": str(r.get("assay_description"))[:400],
                })
            nxt = page["page_meta"].get("next")
            url = ("https://www.ebi.ac.uk" + nxt) if nxt and nxt.startswith("/") else nxt
        time.sleep(0.2)
    return out


def merge(old: list[dict], new: list[dict]) -> tuple[list[dict], int]:
    """Апгрейд assay по сигнатуре, append новых записей целевых доков.

    Дубликаты с одинаковой сигнатурой (doc, type, value, units, relation —
    например два VR 1.55: VSV и Vaccinia) сопоставляются по номеру
    вхождения: относительный порядок записей одной молекулы стабилен между
    скрапами одного и того же API.
    """
    from collections import defaultdict, deque

    sig = lambda r: (r["doc"], r["type"], str(r["value"]), r["units"], r["relation"])
    out = [dict(r) for r in old]
    pool: dict[tuple, deque] = defaultdict(deque)
    for r in new:
        if r["doc"] in TARGET_DOCS:
            pool[sig(r)].append(r)
    upgraded = 0
    for o in out:
        q = pool.get(sig(o))
        if q:
            r = q.popleft()
            if o["assay"] != r["assay"]:
                o["assay"] = r["assay"]
                upgraded += 1
    added = 0
    for q in pool.values():
        for r in q:
            if not any(sig(o) == sig(r) and o["assay"] == r["assay"] for o in out):
                out.append(dict(r))
                added += 1
    return out, upgraded


def main() -> None:
    records = json.load(open(OUT))
    keys = sorted({k for k, rs in records.items()
                   if any(r["doc"] in TARGET_DOCS for r in rs)} | set(SID_MOLS))
    print(f"ключей: {len(keys)}")
    n_app, n_upg = 0, 0
    for k in keys:
        mol_ids = {SID_MOLS[k]} if k in SID_MOLS else set()
        try:
            mol_ids |= set(molecules_for(k))
        except Exception as e:
            print(f"  {k[:16]}: search err {e}")
        if not mol_ids:
            continue
        try:
            acts = activities_for(sorted(mol_ids))
        except Exception as e:
            print(f"  {k[:16]}: activity err {e}")
            continue
        old = records.get(k, [])
        merged, upg = merge(old, acts)
        added = len(merged) - len(old)
        if added or upg:
            records[k] = merged
            n_app += added
            n_upg += upg
            print(f"  {k[:16]}: +{added} записей, {upg} assay обновлено")
        time.sleep(0.2)
    json.dump(records, open(OUT, "w"))
    print(f"→ {OUT}: appends {n_app}, upgrades {n_upg}")


if __name__ == "__main__":
    main()
