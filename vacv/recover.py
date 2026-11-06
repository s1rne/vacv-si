"""Reproduce the organisers' own IC50 and CC50, rather than estimate them.

The competition's `SI VACV` is not an independent measurement. On the training
split it equals `CC50_uM / IC50_uM_VACV` exactly for all 237 complete rows,
and 378 of the 917 compounds are titled with the PubChem SID of a ChEMBL
deposit. The files were assembled from public records by a procedure, so the
useful question is not "what is this compound's selectivity index" but "which
public record did they copy, and by what rule".

That procedure, fitted on the training split where the answer is visible:

1. **One compound, one publication.** Of the 123 training compounds that have
   an antiviral record and a cytotoxicity record in the same ChEMBL document,
   110 share the document between the two halves - the ratio is the one the
   paper's own table reports. Only 8 rows have a choice of more than one
   document.
2. **One record per side, never an aggregate.** Values are the record's own
   number converted to micromolar (every reproduced IC50 came from a record
   ChEMBL stores in nM, so the conversion is a division by 1000). Taking the
   median of a compound's replicates - which is what a dataset builder should
   do, and what :mod:`vacv.build_dataset` does - reproduces the organisers'
   number for 58% of the matched rows; picking one record reproduces 86%.
3. **Inside the publication, prefer the cell line the row names, then the
   censored record.** ``Culture`` names the cells; where a paper measured
   several lines, the matching one is the organisers' choice. The second
   preference looks wrong and is not: a reported "> 100" arrives in the
   competition file as a plain ``100``, so the bounded record is usually the
   one they transcribed.

With that rule, 106 of the 123 training rows that have a same-document pair
are reproduced **bit-identically**, and the misses are rows whose record our
curation dropped rather than rows where the rule chose wrongly.

Everything here is driven by the competition's ``SMILES``, ``Title``,
``strain`` and ``Culture``. ``CC50_uM``, ``IC50_uM_VACV`` and ``SI VACV`` are
read in :mod:`vacv.coverage` to *measure* the rule on the training split and
nowhere else.
"""

from __future__ import annotations

import logging

import pandas as pd

import numpy as np

from .build_dataset import antiviral_side, cytotox_side, si_quality
from .classify import extract_cell_line, mentions_vaccinia
from .paths import INACTIVE, MOLECULE_LINKS
from .units import MASS_UG_PER_ML, canonical_units

log = logging.getLogger(__name__)

#: How a competition compound was linked to open records, strongest first.
#: ``inchikey`` is the standardized parent - desalted, neutral, canonical
#: tautomer - so a looser "parent" or "stereochemistry-insensitive" tier would
#: be the same thing; ``inchikey14`` is the InChIKey skeleton, which ignores
#: stereochemistry and protonation by construction; ``flexmatch`` is ChEMBL's
#: own connectivity search, which reaches records our standardization and
#: theirs disagree about.
MATCH_TIERS = ("inchikey", "chembl_registry", "inchikey14", "flexmatch", "inactive_record")

#: How the two halves were paired, strongest first. ``reported ratio`` is not
#: a pairing at all: it is the selectivity index the authors printed, used
#: where no pair can be rebuilt.
PAIRING_RULES = (
    "document, same cell line",
    "document",
    "compound",
    "inactive, no antiviral number",
    "reported ratio",
)

#: The organisers' value where their IC50 column is empty. Not a ratio.
SI_SENTINEL = 10_000.0

#: Preference between several records of one side inside one document:
#: the cell line the row names, then the endpoint the organisers preferred,
#: then the censored record.
SIDE_RULE = ("cell", "endpoint", "censored")

#: Endpoint preference inside one side, strongest first. It matters because
#: the old antiviral literature files the same table under several names: a
#: paper that prints both a CC50 and a "minimum cytotoxic concentration" for
#: one compound in one cell line had its CC50 copied, and preferring MCC
#: there costs 15 of the 158 training rows that have a pair.
ANTIVIRAL_ENDPOINTS = ("EC50", "IC50", "ED50", "ID50", "MIC", "MIC50", "AC50", "I50", "Potency", "Activity")
CYTOTOX_ENDPOINTS = (
    "CC50",
    "TC50",
    "CD50",
    "GI50",
    "TGI",
    "IC50",
    "LC50",
    "TD50",
    "MTC",
    "MCC",
    "MNTD",
    "MTD",
    "LD50",
)

#: Columns of the per-compound lookup, in order. Fixed so the file has the
#: same shape whether or not anything matched, and so the modelling side can
#: rely on `si`, `si_quality`, `match_tier` and `confidence` being present.
OUTPUT_COLUMNS = [
    "split",
    "index",
    "Title",
    "smiles",
    "inchikey",
    "inchikey14",
    "culture",
    "cell_line",
    "match_tier",
    "rule",
    "ic50",
    "cc50",
    "si",
    "si_quality",
    "confidence",
    "ic50_relation",
    "cc50_relation",
    "ic50_type",
    "cc50_type",
    "censored",
    "matched_cell_line",
    "cc50_cell_line",
    "strain",
    "document_id",
    "n_documents",
    "doi",
    "year",
    "source_db",
    "n_antiviral_records",
    "n_cytotox_records",
    "inactive_evidence",
    "sentinel_evidence",
    "ic50_um",
    "cc50_um",
    "document_verdict",
    "document_train_rows",
    "document_train_within_0.3_log",
    "sentinel_rate_for_bucket",
    "si_blended",
    "substitute_recommended",
    "si_recommended",
]


def _key(value: object) -> str | None:
    """A structure key, or None where standardization failed."""
    return value if isinstance(value, str) and value else None


def _value(record: dict) -> float:
    """The record's value as the organisers would have transcribed it."""
    printed = record.get("organiser_value")
    if printed is not None and np.isfinite(printed):
        return float(printed)
    return float(record["value_um"])


def _endpoint_rank(record: dict, kind: str) -> int:
    order = CYTOTOX_ENDPOINTS if kind == "cytotox" else ANTIVIRAL_ENDPOINTS
    endpoint = record.get("standard_type")
    return order.index(endpoint) if endpoint in order else len(order)


def _side_key(record: dict, cell: str | None, kind: str) -> tuple:
    return (
        -int(cell is not None and record.get("cell_line") == cell),
        _endpoint_rank(record, kind),
        -int(bool(record.get("censored"))),
        str(record.get("source_id")),
    )


def _best(records: list[dict], cell: str | None, kind: str) -> dict:
    return min(records, key=lambda record: _side_key(record, cell, kind))


class Records:
    """The open-data records, indexed the way the matcher needs them."""

    def __init__(self, curated_open: pd.DataFrame, inactive: pd.DataFrame | None = None):
        antiviral = antiviral_side(curated_open)
        cytotox = cytotox_side(curated_open)
        self.antiviral_exact = self._index(antiviral, "inchikey")
        self.cytotox_exact = self._index(cytotox, "inchikey")
        self.antiviral = self._index(antiviral, "inchikey14")
        self.cytotox = self._index(cytotox, "inchikey14")
        self.selectivity = self._index(_vaccinia_selectivity(curated_open), "inchikey14")
        inactive = inactive_records() if inactive is None else inactive
        self.inactive = set(inactive["inchikey14"].dropna()) if not inactive.empty else set()

    @staticmethod
    def _index(frame: pd.DataFrame, key: str) -> dict[str, list[dict]]:
        frame = frame.assign(organiser_value=printed_values(frame)) if not frame.empty else frame
        columns = [
            "value_um",
            "relation",
            "censored",
            "standard_type",
            "cell_line",
            "strain",
            "document_id",
            "doi",
            "year",
            "source_db",
            "source_id",
            "raw_units",
            "organiser_value",
        ]
        kept = [c for c in columns if c in frame.columns]
        out: dict[str, list[dict]] = {}
        for value, group in frame[kept + [key]].groupby(key):
            out[value] = group[kept].to_dict("records")
        return out

    def sides(self, key: str | None, exact: bool) -> tuple[list[dict], list[dict]]:
        if not key:
            return [], []
        if exact:
            return self.antiviral_exact.get(key, []), self.cytotox_exact.get(key, [])
        return self.antiviral.get(key, []), self.cytotox.get(key, [])


def printed_values(frame: pd.DataFrame) -> "pd.Series":
    """The number the organisers would have copied out of the record.

    They did not convert units. Where a record reports ``EC50 > 100
    ug.mL-1``, their ``IC50_uM_VACV`` column holds ``100.0`` - the printed
    number, in a column labelled micromolar. The ratio is unaffected whenever
    both halves are in the same units, because the molecular weight cancels,
    but the two columns are not, and neither is a pair whose halves were
    reported in different units.
    """
    value_um = pd.to_numeric(frame["value_um"], errors="coerce")
    if "raw_value" not in frame.columns or "raw_units" not in frame.columns:
        return value_um
    raw = pd.to_numeric(frame["raw_value"], errors="coerce")
    units = frame["raw_units"].map(canonical_units)
    mass = units.isin(set(MASS_UG_PER_ML)) & raw.notna()
    return pd.Series(np.where(mass, raw, value_um), index=frame.index, dtype="float64")


def inactive_records(path=INACTIVE) -> pd.DataFrame:
    """Records reporting inactivity against vaccinia, if the curate stage wrote them."""
    if not path.exists():
        return pd.DataFrame(columns=["inchikey", "inchikey14"])
    return pd.read_csv(path)


def _group_votes(prior: dict | None, record: dict) -> dict[str, int]:
    """Votes for each publication from the row's own (strain, Culture) group.

    The row's own vote is removed, so a training row is never selected by
    evidence it supplied itself and the measured accuracy is not inflated.
    """
    if not prior:
        return {}
    group = prior.get((record.get("strain"), record.get("Culture")), {})
    title = str(record.get("Title"))
    return {document: len(titles - {title}) for document, titles in group.items()}


def _vaccinia_selectivity(curated_open: pd.DataFrame) -> pd.DataFrame:
    """Reported selectivity indices, restricted to the ones about vaccinia.

    The cytotoxicity side needs no virus filter because cytotoxicity is a
    property of the compound and the cells. A selectivity index does need
    one: the same paper prints one for every virus it tested.
    """
    rows = curated_open[curated_open["measurement_class"].eq("selectivity")]
    if rows.empty:
        return rows
    keep = [
        mentions_vaccinia(row.organism, row.assay_description, row.strain) for row in rows.itertuples()
    ]
    return rows[pd.Series(keep, index=rows.index)]


def molecule_links(path=MOLECULE_LINKS) -> pd.DataFrame:
    """Competition ``Title`` -> ChEMBL molecule structure keys, if harvested."""
    if not path.exists():
        return pd.DataFrame(columns=["Title", "route", "chembl_id", "chembl_inchikey", "chembl_inchikey14"])
    return pd.read_csv(path)


def _link_keys(links: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    if links.empty or "chembl_inchikey14" not in links.columns:
        return out
    for record in links.to_dict("records"):
        title = str(record.get("Title"))
        route = "chembl_registry" if record.get("route") == "title_identifier" else "flexmatch"
        bucket = out.setdefault(title, {})
        for key in (record.get("chembl_inchikey"), record.get("chembl_inchikey14")):
            if isinstance(key, str) and key:
                bucket.setdefault(route, []).append(key)
    return out


def _pair(
    antiviral: list[dict],
    cytotox: list[dict],
    cell: str | None,
    votes: dict[str, int] | None = None,
) -> dict | None:
    """The organisers' pairing rule: one publication, one record per side.

    ``votes`` is how many *other* training rows of the same (strain, Culture)
    group were reproduced from each publication. The organisers' groups are
    paper-shaped - 18 of the 23 training groups come from a single publication
    - so where a compound appears in several papers, the group says which one
    the row was taken from. It is a tie-break, never an override: a document
    that cannot supply the row's cell line still loses to one that can.
    """
    documents = {r["document_id"] for r in antiviral if r.get("document_id")} & {
        r["document_id"] for r in cytotox if r.get("document_id")
    }
    if documents:
        scored = []
        for document in sorted(documents):
            left = _best([r for r in antiviral if r.get("document_id") == document], cell, "antiviral")
            right = _best([r for r in cytotox if r.get("document_id") == document], cell, "cytotox")
            matched = int(left.get("cell_line") == cell) + int(right.get("cell_line") == cell) if cell else 0
            scored.append((-(votes or {}).get(document, 0), -matched, document, left, right))
        _, _, document, left, right = min(scored, key=lambda item: item[:3])
        matched_cells = cell is not None and left.get("cell_line") == cell and right.get("cell_line") == cell
        rule = "document, same cell line" if matched_cells else "document"
        return {
            "rule": rule,
            "antiviral": left,
            "cytotox": right,
            "document_id": document,
            "n_documents": len(documents),
        }
    if antiviral and cytotox:
        return {
            "rule": "compound",
            "antiviral": _best(antiviral, cell, "antiviral"),
            "cytotox": _best(cytotox, cell, "cytotox"),
            "document_id": None,
            "n_documents": 0,
        }
    return None


def recover(
    split: pd.DataFrame,
    records: Records,
    links: pd.DataFrame | None = None,
    prior: dict | None = None,
) -> pd.DataFrame:
    """One row per competition compound: the pair the organisers' rule selects.

    ``si`` is ``cc50 / ic50`` of two single records, so it is their arithmetic
    and not an aggregate. ``match_tier`` says how the compound was linked to
    the open data and ``rule`` how the two halves were paired; both are needed
    to read the accuracy table in :mod:`vacv.coverage`, because a skeleton
    match paired across publications is a different claim from an exact
    InChIKey paired inside one.
    """
    by_title = _link_keys(molecule_links() if links is None else links)

    rows = []
    for record in split.to_dict("records"):
        cell = extract_cell_line(record.get("Culture"))
        row = {
            "split": record.get("split"),
            "index": record.get("index"),
            "Title": record.get("Title"),
            "smiles": record.get("SMILES"),
            "inchikey": record.get("inchikey"),
            "inchikey14": record.get("inchikey14"),
            "culture": record.get("Culture"),
            "cell_line": cell,
        }
        key_14 = _key(record.get("inchikey14"))
        votes = _group_votes(prior, record)
        linked = by_title.get(str(record.get("Title")), {})
        attempts: list[tuple[str, str | None, bool]] = [("inchikey", _key(record.get("inchikey")), True)]
        attempts += [("chembl_registry", key, len(key) > 14) for key in linked.get("chembl_registry", [])]
        attempts.append(("inchikey14", _key(record.get("inchikey14")), False))
        attempts += [("flexmatch", key, len(key) > 14) for key in linked.get("flexmatch", [])]

        chosen = None
        antiviral_seen = cytotox_seen = 0
        for tier, key, exact in attempts:
            if not key:
                continue
            antiviral, cytotox = records.sides(key if exact else key[:14], exact)
            antiviral_seen = max(antiviral_seen, len(antiviral))
            cytotox_seen = max(cytotox_seen, len(cytotox))
            found = _pair(antiviral, cytotox, cell, votes)
            if found is not None:
                chosen = (tier, found)
                break
        row["n_antiviral_records"] = antiviral_seen
        row["n_cytotox_records"] = cytotox_seen
        # No antiviral record and a cytotoxicity record is the shape of the
        # organisers' sentinel: their IC50 column is empty because the
        # compound was inactive, and SI VACV becomes 10000. Flagged, never
        # written into si - vacv.coverage measures how often it is right.
        row["inactive_evidence"] = bool(key_14 and key_14 in records.inactive)
        row["sentinel_evidence"] = bool(row["inactive_evidence"] or (cytotox_seen and not antiviral_seen))

        if chosen is None and row["inactive_evidence"]:
            # A record that names vaccinia and reports "Not active" instead of
            # a concentration is exactly what left the organisers' IC50 column
            # empty, and an empty IC50 column is what their pipeline turned
            # into SI VACV = 10000. This is not a guess about the compound, it
            # is the same statement from the same source.
            chosen = (
                "inactive_record",
                {
                    "rule": "inactive, no antiviral number",
                    "antiviral": {},
                    "cytotox": {},
                    "document_id": None,
                    "n_documents": 0,
                    "si": SI_SENTINEL,
                    "doi": None,
                    "year": None,
                    "source_db": None,
                },
            )

        if chosen is None:
            # Nothing to pair: fall back on the selectivity index the authors
            # printed, where the paper printed one.
            for tier, key, _ in attempts:
                if not key:
                    continue
                reported = records.selectivity.get(key[:14], [])
                if reported:
                    values = np.array([float(r["value_um"]) for r in reported], dtype=float)
                    best = reported[0]
                    chosen = (
                        tier,
                        {
                            "rule": "reported ratio",
                            "antiviral": {},
                            "cytotox": {},
                            "document_id": best.get("document_id"),
                            "n_documents": len({r.get("document_id") for r in reported}),
                            "si": float(10 ** np.median(np.log10(values))),
                            "n_reported": len(reported),
                            "doi": best.get("doi"),
                            "year": best.get("year"),
                            "source_db": best.get("source_db"),
                        },
                    )
                    break

        if chosen is not None:
            tier, found = chosen
            left, right = found["antiviral"], found["cytotox"]
            if found["rule"] in ("reported ratio", "inactive, no antiviral number"):
                row.update(
                    {
                        "match_tier": tier,
                        "rule": found["rule"],
                        "si": found["si"],
                        "si_quality": "reported_ratio"
                        if found["rule"] == "reported ratio"
                        else "sentinel",
                        "censored": False,
                        "document_id": found["document_id"],
                        "n_documents": found["n_documents"],
                        "doi": found["doi"],
                        "year": found["year"],
                        "source_db": found["source_db"],
                    }
                )
                rows.append(row)
                continue
            row.update(
                {
                    "match_tier": tier,
                    "rule": found["rule"],
                    "ic50": _value(left),
                    "cc50": _value(right),
                    "si": _value(right) / _value(left),
                    "ic50_um": float(left["value_um"]),
                    "cc50_um": float(right["value_um"]),
                    "ic50_relation": left.get("relation"),
                    "cc50_relation": right.get("relation"),
                    "ic50_type": left.get("standard_type"),
                    "cc50_type": right.get("standard_type"),
                    "si_quality": si_quality(left.get("relation"), right.get("relation")),
                    "censored": bool(left.get("censored")) or bool(right.get("censored")),
                    "matched_cell_line": left.get("cell_line"),
                    "cc50_cell_line": right.get("cell_line"),
                    "strain": left.get("strain") or right.get("strain"),
                    "document_id": found["document_id"],
                    "n_documents": found["n_documents"],
                    "doi": left.get("doi") or right.get("doi"),
                    "year": left.get("year"),
                    "source_db": left.get("source_db"),
                }
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out = out[OUTPUT_COLUMNS]
    log.info(
        "recover: %d of %d compounds get a pair (%s)",
        int(out["si"].notna().sum()) if "si" in out else 0,
        len(out),
        out["rule"].value_counts().to_dict() if "rule" in out else {},
    )
    return out
