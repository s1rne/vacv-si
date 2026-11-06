"""PubChem BioAssay as a source of VACV antiviral and cytotoxicity data.

Endpoints
    * ``https://eutils.ncbi.nlm.nih.gov/entrez/eutils`` to find assays
      (``pcassay`` has real indexed fields, unlike free-text PUG search);
    * ``https://pubchem.ncbi.nlm.nih.gov/rest/pug`` for assay descriptions,
      data tables and compound SMILES.

Two things decide the shape of this module.

First, the naive query ``vaccinia`` over all assay text returns 3200 assays,
and most are false positives: a modified-vaccinia-Ankara T7 polymerase is a
standard *tool* in Ebola and RSV minigenome assays, so the word appears in
protocols that measure nothing about poxviruses. Restricting to the indexed
assay-name and taxonomy fields (``ANAM``, ``TXNM``) cuts that to ~844.

Second, 827 of those ~844 assays are ChEMBL re-deposits. Harvesting them here
would duplicate :mod:`vacv.sources.chembl` with less metadata (no DOI, no cell
line dictionary), so they are skipped by default and the duplication is
reported rather than hidden. PubChem's genuine contribution is the NCATS qHTS
poxvirus screens, the Broad stage-specific assays, and a handful of
BindingDB-deposited patent assays.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any, Iterator

import pandas as pd

from .. import classify
from ..http import get_json, get_text
from ..paths import raw_dir
from ..schema import conform
from ..units import canonical_relation, to_micromolar

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
SOURCE = "pubchem"

ASSAY_QUERY = "vaccinia[ANAM] OR vaccinia[TXNM]"

#: PubChem result-unit enum -> unit string understood by :mod:`vacv.units`.
UNIT_ENUM = {
    4: "mM",
    5: "uM",
    6: "nM",
    7: "pM",
    8: "fM",
    9: "mg/ml",
    10: "ug/ml",
    11: "ng/ml",
    12: "pg/ml",
    14: "M",
    15: "%",
    16: "ratio",
}

#: Depositors whose vaccinia-tagged records add nothing or belong elsewhere.
#: ChEMBL is harvested directly; the Shanghai records are PDBbind
#: protein-ligand affinities (Kd/Ki against purified enzymes), which cannot
#: take part in a cell-based selectivity index.
SKIP_SOURCES = {
    "ChEMBL",
    "Shanghai Institute of Organic Chemistry",
}

SID_BATCH = 150
CID_BATCH = 150


def _esearch(term: str, db: str = "pcassay", retmax: int = 5000) -> list[str]:
    payload = get_json(
        f"{EUTILS}/esearch.fcgi",
        SOURCE,
        params={"db": db, "term": term, "retmax": retmax, "retmode": "json"},
    )
    return payload["esearchresult"]["idlist"] if payload else []


def _esummary(ids: list[str], db: str = "pcassay") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for start in range(0, len(ids), 200):
        batch = ids[start : start + 200]
        payload = get_json(
            f"{EUTILS}/esummary.fcgi",
            SOURCE,
            params={"db": db, "id": ",".join(batch), "retmode": "json"},
        )
        if not payload:
            continue
        result = payload.get("result", {})
        for uid in result.get("uids", []):
            out[uid] = result[uid]
    return out


def _assay_description(aid: str) -> dict | None:
    payload = get_json(f"{PUG}/assay/aid/{aid}/description/JSON", SOURCE)
    if not payload:
        return None
    return payload["PC_AssayContainer"][0]["assay"]["descr"]


def _active_sids(aid: str) -> list[str]:
    payload = get_json(f"{PUG}/assay/aid/{aid}/sids/JSON", SOURCE, params={"sids_type": "active"})
    if not payload:
        return []
    info = payload.get("InformationList", {}).get("Information", [])
    return [str(sid) for item in info for sid in item.get("SID", [])]


def _assay_rows(aid: str, sids: list[str]) -> Iterator[dict[str, str]]:
    """Full data table for selected SIDs.

    The ``sid=`` filter is honoured by the CSV endpoint (it is silently ignored
    by ``concise``), which is the only way to pull a 300k-row qHTS screen
    without downloading tens of megabytes of inactive wells.
    """
    for start in range(0, len(sids), SID_BATCH):
        batch = sids[start : start + SID_BATCH]
        text = get_text(
            f"{PUG}/assay/aid/{aid}/CSV",
            SOURCE,
            params={"sid": ",".join(batch)},
            suffix=".csv",
        )
        if not text:
            continue
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            # PubChem prefixes every table with RESULT_TYPE and RESULT_DESCR
            # pseudo-rows that carry column metadata, not measurements. Both
            # have an empty SID, so that is the test.
            if not (row.get("PUBCHEM_SID") or "").strip().isdigit():
                continue
            yield row


def _activity_columns(descr: dict) -> list[tuple[str, str | None]]:
    """Columns the depositor marked as the active-concentration readout."""
    columns = []
    for result in descr.get("results") or []:
        if result.get("ac"):
            columns.append((result["name"], UNIT_ENUM.get(result.get("unit"))))
    return columns


def _cid_properties(cids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    unique = sorted({c for c in cids if c and c != "0"})
    for start in range(0, len(unique), CID_BATCH):
        batch = unique[start : start + CID_BATCH]
        text = get_text(
            f"{PUG}/compound/cid/{','.join(batch)}/property/SMILES,MolecularWeight/CSV",
            SOURCE,
            suffix=".csv",
        )
        if not text:
            continue
        for row in csv.DictReader(io.StringIO(text)):
            out[row["CID"]] = row
    return out


def _pmid_dois(pmids: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    unique = sorted({p for p in pmids if p})
    for start in range(0, len(unique), 200):
        batch = unique[start : start + 200]
        payload = get_json(
            f"{EUTILS}/esummary.fcgi",
            SOURCE,
            params={"db": "pubmed", "id": ",".join(batch), "retmode": "json"},
        )
        if not payload:
            continue
        result = payload.get("result", {})
        for uid in result.get("uids", []):
            for article_id in result[uid].get("articleids", []):
                if article_id.get("idtype") == "doi":
                    out[uid] = article_id["value"]
    return out


def harvest(include_chembl_deposits: bool = False, max_active_sids: int = 20000) -> pd.DataFrame:
    aids = _esearch(ASSAY_QUERY)
    summaries = _esummary(aids)
    log.info("pubchem: %d assays match %s", len(aids), ASSAY_QUERY)

    skipped: dict[str, int] = {}
    kept: list[str] = []
    for aid, summary in summaries.items():
        depositor = summary.get("currentsourcename") or "unknown"
        if depositor in SKIP_SOURCES and not (include_chembl_deposits and depositor == "ChEMBL"):
            skipped[depositor] = skipped.get(depositor, 0) + 1
            continue
        kept.append(aid)
    log.info("pubchem: keeping %d assays, skipping %s", len(kept), skipped)

    rows: list[dict] = []
    pmids: list[str] = []
    for aid in sorted(kept, key=int):
        summary = summaries[aid]
        descr = _assay_description(aid)
        if descr is None:
            continue
        activity_columns = _activity_columns(descr)
        if not activity_columns:
            skipped["no_active_concentration_column"] = skipped.get("no_active_concentration_column", 0) + 1
            continue
        sids = _active_sids(aid)
        if not sids:
            skipped["no_active_substances"] = skipped.get("no_active_substances", 0) + 1
            continue
        if len(sids) > max_active_sids:
            log.warning("pubchem: AID %s has %d active SIDs, truncating", aid, len(sids))
            sids = sids[:max_active_sids]

        xrefs = descr.get("xref") or []
        assay_pmids = [str(x["xref"]["pmid"]) for x in xrefs if "pmid" in x.get("xref", {})]
        patents = [str(x["xref"]["patent"]) for x in xrefs if "patent" in x.get("xref", {})]
        pmids.extend(assay_pmids)
        name = descr.get("name") or summary.get("assayname")
        description = " ".join(descr.get("description") or []) or summary.get("assaydescription")
        text = f"{name} {description}"

        for row in _assay_rows(aid, sids):
            for column, units in activity_columns:
                raw = row.get(column)
                if raw in (None, ""):
                    continue
                phenotype = (row.get("Phenotype") or "").strip()
                # BindingDB deposits name the real endpoint in a side column;
                # NCATS calls its AC50 "Potency".
                endpoint = (row.get("Standard Type") or "").strip()
                if not endpoint:
                    endpoint = "AC50" if column.lower().startswith(("potency", "ac50")) else column
                rows.append(
                    {
                        "_cid": row.get("PUBCHEM_CID"),
                        "smiles": row.get("PUBCHEM_EXT_DATASOURCE_SMILES") or None,
                        "standard_type": endpoint,
                        "value_um": None,
                        "raw_value": raw,
                        "raw_units": units,
                        "relation": "=",
                        "assay_description": f"{name} :: {description}"[:2000],
                        # PubChem has no BAO format field; left unknown so that
                        # the pairing filter only rejects formats it is sure of.
                        "assay_format": None,
                        "organism": "Vaccinia virus" if classify.mentions_vaccinia(text) else None,
                        "strain": classify.extract_strain(text),
                        "cell_line": classify.extract_cell_line(summary.get("cellline"), text),
                        "source_db": "PubChem BioAssay",
                        "source_id": f"PUBCHEM_AID{aid}_SID{row.get('PUBCHEM_SID')}_{column}",
                        "assay_id": f"AID{aid}",
                        "document_id": f"PMID{assay_pmids[0]}" if assay_pmids else f"AID{aid}",
                        "doi": None,
                        # A PubChem assay xref PMID is a citation the depositor
                        # attached, not necessarily the paper the numbers come
                        # from: the NCATS poxvirus screens cite a monkeypox
                        # epidemiology paper that contains no chemistry at all.
                        # Recorded, flagged, and never presented as primary.
                        "doi_provenance": "assay_citation" if assay_pmids else None,
                        "_pmid": assay_pmids[0] if assay_pmids else None,
                        "reference": " | ".join(
                            part
                            for part in (
                                f"PubChem AID {aid} ({summary.get('currentsourcename')})",
                                name,
                                "patents: " + ",".join(patents) if patents else "",
                                "cited PMIDs: " + ",".join(assay_pmids) if assay_pmids else "",
                            )
                            if part
                        ),
                        "year": (summary.get("depositdate") or "")[:4] or None,
                        "activity_comment": phenotype or None,
                        "data_validity": row.get("Curve_Description") or None,
                        "_text": text,
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        log.warning("pubchem: no rows harvested")
        return conform(frame)

    properties = _cid_properties(frame["_cid"].dropna().astype(str).tolist())
    dois = _pmid_dois(pmids)

    def resolve(row: pd.Series) -> pd.Series:
        prop = properties.get(str(row["_cid"]), {})
        smiles = row["smiles"] or prop.get("SMILES")
        try:
            mol_weight = float(prop.get("MolecularWeight") or "nan")
        except ValueError:
            mol_weight = float("nan")
        value_um, reason = to_micromolar(
            row["raw_value"], row["raw_units"], mol_weight=mol_weight if mol_weight == mol_weight else None
        )
        cited_doi = dois.get(str(row["_pmid"])) if row["_pmid"] else None
        reference = row["reference"]
        if cited_doi:
            reference = f"{reference} | cited DOI: {cited_doi}"
        return pd.Series(
            {
                "smiles": smiles,
                "value_um": value_um,
                # Deliberately left empty: see the doi_provenance note above.
                # A citation the depositor attached is not the publication of
                # the measurement, and a wrong DOI is worse than none.
                "doi": None,
                "reference": reference,
                "data_validity": row["data_validity"] or (None if reason == "ok" else reason),
            }
        )

    frame[["smiles", "value_um", "doi", "reference", "data_validity"]] = frame.apply(resolve, axis=1)
    frame["measurement_class"] = [
        _classify_pubchem(row) for _, row in frame.iterrows()
    ]

    (raw_dir(SOURCE) / "harvest_summary.json").write_text(
        json.dumps({"assays_matched": len(aids), "assays_used": len(kept), "skipped": skipped, "rows": len(frame)}, indent=2)
    )
    return conform(frame.drop(columns=["_cid", "_pmid", "_text"]))


def _classify_pubchem(row: pd.Series) -> str:
    """PubChem qHTS screens carry a per-compound phenotype call.

    In an infected-cell reporter screen an AC50 whose phenotype is ``cytotoxic``
    is a cytotoxicity concentration, not an antiviral potency, even though it
    sits in the same column of the same assay. Trusting the column alone would
    file host-cell killers as antivirals and invert their selectivity index.
    """
    comment = row.get("activity_comment")
    phenotype = str(comment).strip().lower() if isinstance(comment, str) else ""
    if phenotype in {"cytotoxic", "cytotoxicity", "toxic"}:
        return "cytotox"
    if phenotype in {"activator", "fluorescent", "inactive", "inconclusive"}:
        return "unknown"
    return classify.classify(
        row.get("standard_type"),
        description=row.get("assay_description"),
        organism=row.get("organism"),
    )
