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


class TooManyResults(RuntimeError):
    """A filtered query matched far more rows than the filter can explain."""


def _paged(
    resource: str,
    key: str,
    params: dict[str, Any],
    *,
    refresh: bool = False,
    max_total: int | None = None,
) -> Iterator[dict]:
    offset = 0
    while True:
        query = dict(params, limit=PAGE, offset=offset)
        payload = get_json(f"{BASE}/{resource}.json", SOURCE, params=query, refresh=refresh)
        if not payload:
            return
        total = payload["page_meta"]["total_count"]
        if max_total is not None and total > max_total and offset == 0:
            raise TooManyResults(f"{resource} filter {params} matched {total} rows (cap {max_total})")
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


ID_FIELD = {"assay": "assay_chembl_id", "document": "document_chembl_id", "cell_line": "cell_chembl_id"}


def _fetch_by_id(resource: str, key: str, ids: Iterable[str], refresh: bool = False) -> dict[str, dict]:
    field = ID_FIELD[resource]
    out: dict[str, dict] = {}
    for batch in _batched(sorted(set(ids))):
        for item in _paged(resource, key, {f"{field}__in": ",".join(batch)}, refresh=refresh):
            out[item[field]] = item
    return out


def _vaccinia_activities(refresh: bool = False) -> tuple[dict[int, dict], dict[str, dict]]:
    """Everything ChEMBL ties to vaccinia directly. Bounded and precise."""
    activities: dict[int, dict] = {}
    seed_assays = {a["assay_chembl_id"]: a for a in vaccinia_assays(refresh)}

    for target in vaccinia_targets(refresh):
        for activity in _paged(
            "activity", "activities", {"target_chembl_id": target["target_chembl_id"]}, refresh=refresh
        ):
            activities[activity["activity_id"]] = activity

    for batch in _batched(sorted(seed_assays)):
        for activity in _paged(
            "activity", "activities", {"assay_chembl_id__in": ",".join(batch)}, refresh=refresh
        ):
            activities[activity["activity_id"]] = activity

    log.info("chembl: %d activities tied to vaccinia directly", len(activities))
    return activities, seed_assays


def expandable_documents(documents: dict[str, dict]) -> list[str]:
    """Documents it is safe and meaningful to pull in their entirety.

    ChEMBL's notion of a document includes depositor containers, not only
    papers: ``CHEMBL1201862`` is titled "PubChem BioAssay data set", has
    ``doc_type == "DATASET"``, and owns 7.4 million activities from every
    depositor PubChem ever had. Expanding it would download most of ChEMBL,
    and - worse - treating it as "the same publication" would pair a vaccinia
    IC50 with a cytotoxicity measurement from an unrelated deposit, which is
    exactly the failure mode the strict pairing rule exists to prevent.

    So only real publications are expanded, and only they are allowed to act
    as a pairing key (see :func:`_row`).
    """
    return [
        chembl_id
        for chembl_id, document in documents.items()
        if document.get("doc_type") == "PUBLICATION" and document.get("src_id") == 1
    ]


#: A single publication with more activities than this is a bulk deposit
#: wearing a paper's clothes; skip it rather than download it.
MAX_ACTIVITIES_PER_DOCUMENT = 20_000


def harvest(refresh: bool = False) -> pd.DataFrame:
    activities, seed_assays = _vaccinia_activities(refresh)

    document_ids = {a["document_chembl_id"] for a in activities.values() if a.get("document_chembl_id")}
    docs = _fetch_by_id("document", "documents", document_ids, refresh)
    expandable = expandable_documents(docs)
    log.info(
        "chembl: %d documents, %d are publications and will be expanded",
        len(docs),
        len(expandable),
    )

    skipped_documents: list[str] = []
    for batch in _batched(sorted(expandable)):
        try:
            found = list(
                _paged(
                    "activity",
                    "activities",
                    {"document_chembl_id__in": ",".join(batch)},
                    refresh=refresh,
                    max_total=MAX_ACTIVITIES_PER_DOCUMENT * len(batch),
                )
            )
        except TooManyResults as exc:
            # Fall back to one document at a time so a single bulk deposit
            # costs that document, not the whole batch.
            log.warning("chembl: %s; retrying document by document", exc)
            found = []
            for chembl_id in batch:
                try:
                    found.extend(
                        _paged(
                            "activity",
                            "activities",
                            {"document_chembl_id": chembl_id},
                            refresh=refresh,
                            max_total=MAX_ACTIVITIES_PER_DOCUMENT,
                        )
                    )
                except TooManyResults as inner:
                    log.warning("chembl: skipping %s: %s", chembl_id, inner)
                    skipped_documents.append(chembl_id)
        for activity in found:
            activities.setdefault(activity["activity_id"], activity)
    log.info("chembl: %d activities after document widening", len(activities))

    records = list(activities.values())
    assay_ids = {a["assay_chembl_id"] for a in records if a.get("assay_chembl_id")}
    assays = dict(seed_assays)
    assays.update(_fetch_by_id("assay", "assays", assay_ids, refresh))

    # Expansion brings in activities from documents not seen in the first pass.
    missing_docs = {
        a["document_chembl_id"]
        for a in records
        if a.get("document_chembl_id") and a["document_chembl_id"] not in docs
    }
    docs.update(_fetch_by_id("document", "documents", missing_docs, refresh))

    cell_ids = {a.get("cell_chembl_id") for a in assays.values() if a.get("cell_chembl_id")}
    cells = _fetch_by_id("cell_line", "cell_lines", cell_ids, refresh)

    (raw_dir(SOURCE) / "harvest_summary.json").write_text(
        json.dumps(
            {
                "documents": len(docs),
                "documents_expanded": len(expandable),
                "documents_skipped_as_bulk_deposits": skipped_documents,
                "activities": len(records),
                "assays": len(assays),
                "cell_lines": len(cells),
            },
            indent=2,
        )
    )

    rows = [_row(activity, assays, docs, cells) for activity in records]
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
        # Only a real publication may act as a pairing key. A depositor
        # container shared by thousands of unrelated assays would make the
        # strict rule meaningless, so it is left empty and such rows can only
        # be paired at the weakest "compound" level.
        "document_id": activity.get("document_chembl_id")
        if document.get("doc_type") == "PUBLICATION"
        else None,
        "doi": document.get("doi") or document.get("doi_chembl"),
        "doi_provenance": "document" if (document.get("doi") or document.get("doi_chembl")) else None,
        "reference": reference or None,
        "year": activity.get("document_year") or document.get("year"),
        "activity_comment": activity.get("activity_comment"),
        "data_validity": activity.get("data_validity_comment") or (None if reason.startswith("ok") else reason),
    }


def is_biochemical(assay_format: str | None) -> bool:
    return (assay_format or "") in BIOCHEMICAL_FORMATS
