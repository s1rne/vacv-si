"""Resolve the competition's ``Title`` identifiers to database records.

378 of the 917 competition rows are titled with a PubChem SID
(``sid103460068``) and four with a ChEMBL ID (``CHEMBL6102784``). A PubChem
substance is a *depositor's* record, so its xref carries the depositor's own
identifier: every one of these SIDs was deposited by ChEMBL and resolves to a
``CHEMBL…`` molecule ID in one call.

That is a stronger link than any structure match. It says which database
record the organisers took the compound from, so the activities of that record
are the organisers' own numbers rather than a molecule that happens to have
the same skeleton.

    GET /rest/pug/substance/sid/<list>/xrefs/RegistryID,SourceName/JSON

Licence: PubChem imposes no restrictions on use or redistribution.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..competition import identifiers as competition_identifiers
from ..http import get_json

log = logging.getLogger(__name__)

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
SOURCE = "identifiers"

#: SIDs per request. The URL stays well inside PUG REST's length limit and
#: 378 substances resolve in eight calls.
BATCH = 50


def _sid_registry(sids: list[str], refresh: bool = False) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for start in range(0, len(sids), BATCH):
        batch = sids[start : start + BATCH]
        payload = get_json(
            f"{PUG}/substance/sid/{','.join(batch)}/xrefs/RegistryID,SourceName/JSON",
            SOURCE,
            refresh=refresh,
        )
        if not payload:
            continue
        for item in payload.get("InformationList", {}).get("Information", []):
            registry = item.get("RegistryID") or []
            sources = item.get("SourceName") or []
            if registry:
                out[str(item["SID"])] = (registry[0], sources[0] if sources else "")
    return out


def competition_chembl_ids(refresh: bool = False) -> pd.DataFrame:
    """``Title`` -> ChEMBL molecule ID, for the titles that are identifiers."""
    table = competition_identifiers()
    if table.empty:
        return pd.DataFrame(columns=["Title", "kind", "chembl_id", "inchikey", "inchikey14"])

    rows = []
    chembl_titles = table[table["kind"].eq("chembl")]
    for record in chembl_titles.to_dict("records"):
        rows.append({**record, "chembl_id": record["value"], "depositor": "ChEMBL"})

    sids = sorted(set(table.loc[table["kind"].eq("sid"), "value"]))
    resolved = _sid_registry(sids, refresh=refresh) if sids else {}
    for record in table[table["kind"].eq("sid")].to_dict("records"):
        found = resolved.get(record["value"])
        if not found or not found[0].upper().startswith("CHEMBL"):
            continue
        rows.append({**record, "chembl_id": found[0].upper(), "depositor": found[1]})

    out = pd.DataFrame(rows)
    log.info(
        "identifiers: %d of %d titled rows resolve to a ChEMBL molecule",
        len(out),
        len(table),
    )
    return out
