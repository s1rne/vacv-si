"""ChEMBL as a source of VACV antiviral and matching cytotoxicity data.

Endpoint: ``https://www.ebi.ac.uk/chembl/api/data`` (REST, no key, CC BY-SA 3.0).

The harvest runs in two widenings, because the cytotoxicity half of a
selectivity index is almost never filed under a virus target:

1. everything attached to a vaccinia **target** (``organism__icontains=vaccinia``)
   and everything in an **assay** whose description mentions vaccinia;
2. every activity in the **documents** those hits belong to. A 1990 paper that
   reports "ED50 against vaccinia virus" also reports "CC50 in Vero cells", but
   the latter is filed against a human cell-line target and would be invisible
   to a virus-only query.

Widening 2 is what makes same-publication pairing possible at all. It also
drags in unrelated measurements from the same papers, which is why
:mod:`vacv.classify` and :mod:`vacv.build_dataset` filter rather than trust.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Iterator

import pandas as pd

from .. import classify
from ..http import get_json
from ..paths import raw_dir
from ..schema import conform
from ..units import canonical_relation, from_pchembl, to_micromolar

log = logging.getLogger(__name__)

BASE = "https://www.ebi.ac.uk/chembl/api/data"
SOURCE = "chembl"
PAGE = 1000
IN_BATCH = 25

#: BAO formats that cannot yield a cell-based potency or cytotoxicity.
BIOCHEMICAL_FORMATS = {
    "single protein format",
    "protein complex format",
    "protein format",
    "nucleic acid format",
    "small-molecule physicochemical format",
    "microsome format",
    "subcellular format",
    "cell-free format",
}


def _paged(resource: str, key: str, params: dict[str, Any], *, refresh: bool = False) -> Iterator[dict]:
    offset = 0
    total = None
    while True:
        query = dict(params, limit=PAGE, offset=offset)
        payload = get_json(f"{BASE}/{resource}.json", SOURCE, params=query, refresh=refresh)
        if not payload:
            return
        total = payload["page_meta"]["total_count"]
        items = payload.get(key) or []
        yield from items
        offset += PAGE
        if offset >= total or not items:
            return


def _batched(values: Iterable[str], size: int = IN_BATCH) -> Iterator[list[str]]:
    chunk: list[str] = []
    for value in values:
        if value:
            chunk.append(value)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def vaccinia_targets(refresh: bool = False) -> list[dict]:
    targets = list(_paged("target", "targets", {"organism__icontains": "vaccinia"}, refresh=refresh))
    log.info("chembl: %d vaccinia targets", len(targets))
    return targets


def vaccinia_assays(refresh: bool = False) -> list[dict]:
    """Assays whose free-text description mentions vaccinia.

    ``assay/search`` is the Lucene-backed endpoint; it is the only way to reach
    assays whose target is a cell line but whose description names the virus.
    """
    assays: dict[str, dict] = {}
    offset = 0
    while True:
        payload = get_json(
            f"{BASE}/assay/search.json",
            SOURCE,
            params={"q": "vaccinia", "limit": 100, "offset": offset},
            refresh=refresh,
        )
        if not payload:
            break
        for assay in payload.get("assays") or []:
            assays[assay["assay_chembl_id"]] = assay
        offset += 100
        if offset >= payload["page_meta"]["total_count"]:
            break
    log.info("chembl: %d assays mention vaccinia", len(assays))
    return list(assays.values())


def _documents_of_interest(refresh: bool = False) -> tuple[set[str], dict[str, dict]]:
    documents: set[str] = set()
    seed_assays = {a["assay_chembl_id"]: a for a in vaccinia_assays(refresh)}
    for assay in seed_assays.values():
        if assay.get("document_chembl_id"):
            documents.add(assay["document_chembl_id"])

    for target in vaccinia_targets(refresh):
        for activity in _paged(
            "activity",
            "activities",
            {
                "target_chembl_id": target["target_chembl_id"],
                "only": "document_chembl_id,assay_chembl_id",
            },
            refresh=refresh,
        ):
            if activity.get("document_chembl_id"):
                documents.add(activity["document_chembl_id"])
    log.info("chembl: %d documents of interest", len(documents))
    return documents, seed_assays


ID_FIELD = {"assay": "assay_chembl_id", "document": "document_chembl_id", "cell_line": "cell_chembl_id"}


def _fetch_by_id(resource: str, key: str, ids: Iterable[str], refresh: bool = False) -> dict[str, dict]:
    field = ID_FIELD[resource]
    out: dict[str, dict] = {}
    for batch in _batched(sorted(set(ids))):
        for item in _paged(resource, key, {f"{field}__in": ",".join(batch)}, refresh=refresh):
            out[item[field]] = item
    return out


def harvest(refresh: bool = False) -> pd.DataFrame:
    documents, seed_assays = _documents_of_interest(refresh)

    activities: list[dict] = []
    for batch in _batched(sorted(documents)):
        activities.extend(
            _paged("activity", "activities", {"document_chembl_id__in": ",".join(batch)}, refresh=refresh)
        )
    log.info("chembl: %d activities in those documents", len(activities))

    assay_ids = {a["assay_chembl_id"] for a in activities if a.get("assay_chembl_id")}
    assays = dict(seed_assays)
    assays.update(_fetch_by_id("assay", "assays", assay_ids, refresh))

    docs = _fetch_by_id("document", "documents", documents, refresh)

    cell_ids = {a.get("cell_chembl_id") for a in assays.values() if a.get("cell_chembl_id")}
    cells = _fetch_by_id("cell_line", "cell_lines", cell_ids, refresh)

    raw = raw_dir(SOURCE) / "harvest_summary.json"
    raw.write_text(
        json.dumps(
            {
                "documents": len(documents),
                "activities": len(activities),
                "assays": len(assays),
                "cell_lines": len(cells),
            },
            indent=2,
        )
    )

    rows = [_row(activity, assays, docs, cells) for activity in activities]
    return conform(pd.DataFrame(rows))


def _row(activity: dict, assays: dict[str, dict], docs: dict[str, dict], cells: dict[str, dict]) -> dict:
    assay = assays.get(activity.get("assay_chembl_id") or "", {})
    document = docs.get(activity.get("document_chembl_id") or "", {})
    cell = cells.get(assay.get("cell_chembl_id") or "", {})

    description = activity.get("assay_description") or assay.get("description")
    organism = activity.get("target_organism") or assay.get("assay_organism")
    cell_text = " ".join(
        str(part)
        for part in (
            cell.get("cell_name"),
            cell.get("cell_description"),
            assay.get("assay_cell_type"),
            activity.get("target_pref_name"),
        )
        if part
    )

    standard_type = activity.get("standard_type") or activity.get("type")
    value_um, reason = to_micromolar(
        activity.get("standard_value"),
        activity.get("standard_units"),
        mol_weight=None,
    )
    if value_um is None and reason in {"no_units", "mass_units_without_mw"}:
        # pChEMBL is already unit-normalised by ChEMBL itself and survives the
        # mass-concentration case, because ChEMBL did the MW conversion.
        value_um = from_pchembl(activity.get("pchembl_value"))
        if value_um is not None:
            reason = "ok_via_pchembl"

    reference = " ".join(
        str(part)
        for part in (
            document.get("journal"),
            document.get("year"),
            document.get("volume"),
            document.get("first_page"),
            document.get("title"),
        )
        if part
    )

    bao_label = (activity.get("bao_label") or "").lower()

    return {
        "smiles": activity.get("canonical_smiles"),
        "standard_type": standard_type,
        "measurement_class": classify.classify(
            standard_type,
            description=description,
            organism=organism,
            assay_type=assay.get("assay_type") or activity.get("assay_type"),
        ),
        "value_um": value_um,
        "relation": canonical_relation(activity.get("standard_relation") or activity.get("relation")),
        "raw_value": activity.get("standard_value"),
        "raw_units": activity.get("standard_units"),
        "assay_description": description,
        "assay_format": bao_label or None,
        "organism": organism,
        "strain": classify.extract_strain(assay.get("assay_strain"), organism, description),
        "cell_line": classify.extract_cell_line(cell_text, description),
        "source_db": "ChEMBL",
        "source_id": f"CHEMBL_ACT_{activity.get('activity_id')}",
        "assay_id": activity.get("assay_chembl_id"),
        "document_id": activity.get("document_chembl_id"),
        "doi": document.get("doi") or document.get("doi_chembl"),
        "doi_provenance": "document" if (document.get("doi") or document.get("doi_chembl")) else None,
        "reference": reference or None,
        "year": activity.get("document_year") or document.get("year"),
        "activity_comment": activity.get("activity_comment"),
        "data_validity": activity.get("data_validity_comment") or (None if reason.startswith("ok") else reason),
    }


def is_biochemical(assay_format: str | None) -> bool:
    return (assay_format or "") in BIOCHEMICAL_FORMATS
