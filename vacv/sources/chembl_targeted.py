"""ChEMBL again, this time keyed on the competition compounds themselves.

:mod:`vacv.sources.chembl` starts from the word "vaccinia" and walks outwards:
vaccinia targets, assays whose text names the virus, then every activity in
the publications those hits belong to. That finds the poxvirus literature
ChEMBL has indexed *under that word*, and it is how the corpus gets its
antiviral half.

It systematically misses two things, and both of them cost coverage:

1. **The other half of a known compound.** A competition compound can have a
   vaccinia EC50 from a paper ChEMBL indexed and its CC50 in a *different*
   paper - a cytotoxicity screen, a later SAR study - that never says
   "vaccinia" and is therefore invisible to the first harvest. Without the
   CC50 there is no ratio and the compound stays uncovered.
2. **The rest of the table.** A paper reports a series; ChEMBL indexes the
   paper; the first harvest reaches it only if one of *its* assays matched the
   query. Papers whose assay text says "cowpox" or "orthopoxvirus" in the
   indexed field, or whose vaccinia assay ChEMBL filed against a cell-line
   target with a terse description, drop out with all their compounds.

So this module starts from the molecules instead:

* resolve every competition compound to ChEMBL molecule IDs - by the
  identifier in its ``Title`` (:mod:`vacv.sources.identifiers`), by exact
  standardized InChIKey, and by ``flexmatch`` on the standardized SMILES,
  which is ChEMBL's own connectivity search and returns salts, stereoisomers
  and tautomers of the same structure;
* enumerate **all** their activities cheaply with ``only=``, then download the
  full records of the ones that can be an IC50 or a CC50;
* expand, in full, every publication among them that is about poxviruses -
  every compound, every activity - so that a paper's whole series arrives
  together.

Only ``SMILES`` and ``Title`` of the competition file are used, through
:mod:`vacv.competition`. The harvester returns an empty frame when the file is
absent, so the open dataset still builds without it.

Endpoint and licence are those of :mod:`vacv.sources.chembl`: ChEMBL REST, no
key, data CC BY-SA 3.0.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable, Iterator

import pandas as pd

from .. import classify, competition
from ..http import get_json
from ..paths import MOLECULE_LINKS, raw_dir
from ..schema import conform
from . import chembl, identifiers

log = logging.getLogger(__name__)

BASE = chembl.BASE
SOURCE = "chembl_targeted"
PAGE = 1000
IN_BATCH = 25

#: Endpoint names that can be one half of a selectivity index. Anything else
#: (binding constants, permeability, half-lives) is not worth a full record.
USABLE_TYPES = {t.upper() for t in classify.CYTOTOX_TYPES | classify.AMBIGUOUS_TYPES}

#: Enumeration is cheap; full records are not. A molecule with more usable
#: activities than this is a screening-library entry whose data is already in
#: the main harvest if it is about poxviruses at all.
MAX_USABLE_PER_MOLECULE = 2_000

#: Documents to expand in full. The filter below keeps only poxvirus papers,
#: and 400 is far above what it has ever returned; it is a stop, not a target.
MAX_DOCUMENTS = 400

_POX_TEXT = ("vaccinia", "vacv", "poxvir", "orthopox", "cowpox", "variola", "monkeypox", "ectromelia")


def _paged(resource: str, key: str, params: dict, *, refresh: bool = False) -> Iterator[dict]:
    offset = 0
    while True:
        payload = get_json(
            f"{BASE}/{resource}.json", SOURCE, params=dict(params, limit=PAGE, offset=offset), refresh=refresh
        )
        if not payload:
            return
        items = payload.get(key) or []
        yield from items
        total = payload["page_meta"]["total_count"]
        offset += PAGE
        if offset >= total or not items:
            return


def _batched(values: Iterable[str], size: int = IN_BATCH) -> Iterator[list[str]]:
    chunk: list[str] = []
    for value in values:
        if value:
            chunk.append(str(value))
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def molecules_by_inchikey(keys: Iterable[str], refresh: bool = False) -> dict[str, dict]:
    """``inchikey -> molecule record`` for the keys ChEMBL holds exactly."""
    out: dict[str, dict] = {}
    for batch in _batched(sorted(set(keys))):
        for molecule in _paged(
            "molecule",
            "molecules",
            {"molecule_structures__standard_inchi_key__in": ",".join(batch)},
            refresh=refresh,
        ):
            structures = molecule.get("molecule_structures") or {}
            key = structures.get("standard_inchi_key")
            if key:
                out[key] = molecule
    return out


def molecules_by_id(ids: Iterable[str], refresh: bool = False) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for batch in _batched(sorted(set(ids))):
        for molecule in _paged(
            "molecule", "molecules", {"molecule_chembl_id__in": ",".join(batch)}, refresh=refresh
        ):
            out[molecule["molecule_chembl_id"]] = molecule
    return out


def molecules_by_flexmatch(smiles: str, refresh: bool = False) -> list[dict]:
    """ChEMBL's own connectivity search: salts, stereoisomers, tautomers.

    This is the looser-than-exact structure match done on the database's side
    rather than ours, which matters because ChEMBL standardizes differently
    from :mod:`vacv.curate`: a molecule stored as a hydrochloride under one
    InChIKey is reachable from the free base's SMILES only this way.
    """
    payload = get_json(
        f"{BASE}/molecule.json",
        SOURCE,
        params={"molecule_structures__canonical_smiles__flexmatch": smiles, "limit": 50},
        refresh=refresh,
    )
    if not payload:
        return []
    return list(payload.get("molecules") or [])


def competition_molecules(refresh: bool = False) -> pd.DataFrame:
    """Every ChEMBL molecule a competition compound resolves to, and how.

    The structure ChEMBL stores travels with the link, standardized by
    :mod:`vacv.curate` so that it is comparable with the keys of the curated
    corpus. That is what lets the identifier route find records our own
    structure match misses: the organisers' ``sid…`` title points at a ChEMBL
    molecule whose stored structure is a salt or a different tautomer of the
    SMILES they printed.
    """
    compounds = competition.all_compounds()
    if compounds.empty:
        return pd.DataFrame(columns=["Title", "inchikey", "inchikey14", "chembl_id", "route"])

    rows: list[dict] = []
    resolved_titles: set[str] = set()
    molecules: dict[str, dict] = {}

    by_identifier = identifiers.competition_chembl_ids(refresh=refresh)
    if not by_identifier.empty:
        molecules.update(molecules_by_id(by_identifier["chembl_id"], refresh=refresh))
    for record in by_identifier.to_dict("records"):
        rows.append(
            {
                "Title": record["Title"],
                "inchikey": record.get("inchikey"),
                "inchikey14": record.get("inchikey14"),
                "chembl_id": record["chembl_id"],
                "route": "title_identifier",
            }
        )
        resolved_titles.add(str(record["Title"]))

    exact = molecules_by_inchikey(compounds["inchikey"].dropna(), refresh=refresh)
    for record in compounds.to_dict("records"):
        found = exact.get(record.get("inchikey"))
        if found:
            molecules[found["molecule_chembl_id"]] = found
            rows.append(
                {
                    "Title": record.get("Title"),
                    "inchikey": record.get("inchikey"),
                    "inchikey14": record.get("inchikey14"),
                    "chembl_id": found["molecule_chembl_id"],
                    "route": "inchikey",
                }
            )
            resolved_titles.add(str(record.get("Title")))
    log.info("targeted: %d of %d compounds resolve by exact InChIKey", len(exact), len(compounds))

    # flexmatch is one request per compound, so it is spent only on the
    # compounds nothing cheaper has resolved.
    pending = [
        record
        for record in compounds.to_dict("records")
        if str(record.get("Title")) not in resolved_titles and isinstance(record.get("smiles_std"), str)
    ]
    flex = 0
    for record in pending:
        for molecule in molecules_by_flexmatch(record["smiles_std"], refresh=refresh):
            molecules[molecule["molecule_chembl_id"]] = molecule
            rows.append(
                {
                    "Title": record.get("Title"),
                    "inchikey": record.get("inchikey"),
                    "inchikey14": record.get("inchikey14"),
                    "chembl_id": molecule["molecule_chembl_id"],
                    "route": "flexmatch",
                }
            )
            flex += 1
    log.info("targeted: flexmatch searched %d compounds and found %d molecules", len(pending), flex)

    out = pd.DataFrame(rows).drop_duplicates()
    out = _attach_structures(out, molecules)
    log.info("targeted: %d distinct ChEMBL molecules for the competition set", out["chembl_id"].nunique())
    return out


def _attach_structures(links: pd.DataFrame, molecules: dict[str, dict]) -> pd.DataFrame:
    """Standardize each linked ChEMBL structure the way the corpus is standardized."""
    from ..curate import standardize

    smiles, keys, skeletons = [], [], []
    for chembl_id in links["chembl_id"]:
        structures = (molecules.get(chembl_id) or {}).get("molecule_structures") or {}
        reported = structures.get("canonical_smiles")
        standardized = standardize(reported) if reported else None
        smiles.append(reported)
        keys.append(standardized.inchikey if standardized else None)
        skeletons.append(standardized.inchikey[:14] if standardized and standardized.inchikey else None)
    out = links.copy()
    out["chembl_smiles"] = smiles
    out["chembl_inchikey"] = keys
    out["chembl_inchikey14"] = skeletons
    return out


def _enumerate_activities(molecule_ids: Iterable[str], refresh: bool = False) -> pd.DataFrame:
    """Light index of every activity of those molecules.

    ``only=`` keeps the payload to the six fields needed to decide whether an
    activity is worth a full record. Without it this step downloads tens of
    megabytes of assay prose for molecules that turn out to be irrelevant.
    """
    fields = "activity_id,document_chembl_id,assay_chembl_id,molecule_chembl_id,standard_type,assay_description"
    rows: list[dict] = []
    for batch in _batched(sorted(set(molecule_ids))):
        for activity in _paged(
            "activity",
            "activities",
            {"molecule_chembl_id__in": ",".join(batch), "only": fields},
            refresh=refresh,
        ):
            rows.append(activity)
    frame = pd.DataFrame(rows)
    log.info("targeted: %d activities indexed for the competition molecules", len(frame))
    return frame


def _usable(index: pd.DataFrame) -> pd.DataFrame:
    if index.empty:
        return index
    types = index["standard_type"].fillna("").astype(str).str.replace(" ", "").str.upper()
    keep = types.isin(USABLE_TYPES)
    usable = index[keep]
    counts = usable.groupby("molecule_chembl_id").size()
    noisy = set(counts[counts > MAX_USABLE_PER_MOLECULE].index)
    if noisy:
        log.info("targeted: skipping %d molecules with more than %d usable activities", len(noisy), MAX_USABLE_PER_MOLECULE)
        usable = usable[~usable["molecule_chembl_id"].isin(noisy)]
    log.info("targeted: %d of %d indexed activities can be an IC50 or a CC50", len(usable), len(index))
    return usable


def _pox_documents(index: pd.DataFrame, documents: dict[str, dict]) -> list[str]:
    """Publications about poxviruses among those holding a competition compound.

    Two independent reads of the same question: does any activity of ours in
    that document name a poxvirus, or does the document's own title or
    abstract. The second catches papers whose assay descriptions are terse
    ("inhibition of virus replication in Vero cells") and the first catches
    papers whose abstract is about something else entirely.
    """
    text_hits: set[str] = set()
    if not index.empty:
        described = index["assay_description"].fillna("").astype(str).str.lower()
        mask = described.apply(lambda text: any(word in text for word in _POX_TEXT))
        text_hits = set(index.loc[mask, "document_chembl_id"].dropna())

    chosen = []
    for chembl_id, document in documents.items():
        if document.get("doc_type") != "PUBLICATION" or document.get("src_id") != 1:
            continue
        prose = " ".join(
            str(document.get(field) or "") for field in ("title", "abstract", "journal")
        ).lower()
        if chembl_id in text_hits or any(word in prose for word in _POX_TEXT):
            chosen.append(chembl_id)
    log.info("targeted: %d of %d documents are poxvirus publications", len(chosen), len(documents))
    return sorted(chosen)[:MAX_DOCUMENTS]


def _full_activities(activity_ids: Iterable[int], refresh: bool = False) -> list[dict]:
    out: list[dict] = []
    for batch in _batched([str(a) for a in activity_ids]):
        out.extend(_paged("activity", "activities", {"activity_id__in": ",".join(batch)}, refresh=refresh))
    return out


def harvest(refresh: bool = False) -> pd.DataFrame:
    molecules = competition_molecules(refresh=refresh)
    if molecules.empty:
        log.warning("targeted: no competition file in data/raw/kaggle, nothing to look up")
        return conform(pd.DataFrame())

    MOLECULE_LINKS.parent.mkdir(parents=True, exist_ok=True)
    molecules.to_csv(MOLECULE_LINKS, index=False)
    log.info("targeted: wrote %s", MOLECULE_LINKS)

    index = _enumerate_activities(molecules["chembl_id"], refresh=refresh)
    usable = _usable(index)

    document_ids = set(index["document_chembl_id"].dropna()) if not index.empty else set()
    documents = chembl._fetch_by_id("document", "documents", document_ids, refresh)
    expandable = _pox_documents(index, documents)

    activities: dict[int, dict] = {}
    for activity in _full_activities(usable["activity_id"], refresh=refresh):
        activities[activity["activity_id"]] = activity
    log.info("targeted: %d full activity records for the competition molecules", len(activities))

    for batch in _batched(expandable):
        for activity in _paged(
            "activity", "activities", {"document_chembl_id__in": ",".join(batch)}, refresh=refresh
        ):
            activities.setdefault(activity["activity_id"], activity)
    log.info("targeted: %d activities after expanding %d poxvirus publications", len(activities), len(expandable))

    records = list(activities.values())
    assay_ids = {a.get("assay_chembl_id") for a in records if a.get("assay_chembl_id")}
    assays = chembl._fetch_by_id("assay", "assays", assay_ids, refresh)
    missing_docs = {
        a["document_chembl_id"]
        for a in records
        if a.get("document_chembl_id") and a["document_chembl_id"] not in documents
    }
    documents.update(chembl._fetch_by_id("document", "documents", missing_docs, refresh))
    cell_ids = {a.get("cell_chembl_id") for a in assays.values() if a.get("cell_chembl_id")}
    cells = chembl._fetch_by_id("cell_line", "cell_lines", cell_ids, refresh)

    (raw_dir(SOURCE) / "harvest_summary.json").write_text(
        json.dumps(
            {
                "competition_molecules": int(molecules["chembl_id"].nunique()),
                "resolution_routes": molecules.groupby("route")["chembl_id"].nunique().to_dict(),
                "activities_indexed": int(len(index)),
                "activities_usable": int(len(usable)),
                "documents_seen": len(documents),
                "documents_expanded": len(expandable),
                "activities": len(records),
            },
            indent=2,
        )
    )

    rows = [chembl._row(activity, assays, documents, cells) for activity in records]
    return conform(pd.DataFrame(rows))
