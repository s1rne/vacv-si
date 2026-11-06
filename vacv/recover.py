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

from .build_dataset import antiviral_side, cytotox_side, si_quality
from .classify import extract_cell_line
from .paths import MOLECULE_LINKS

log = logging.getLogger(__name__)

#: How a competition compound was linked to open records, strongest first.
#: ``inchikey`` is the standardized parent - desalted, neutral, canonical
#: tautomer - so a looser "parent" or "stereochemistry-insensitive" tier would
#: be the same thing; ``inchikey14`` is the InChIKey skeleton, which ignores
#: stereochemistry and protonation by construction; ``flexmatch`` is ChEMBL's
#: own connectivity search, which reaches records our standardization and
#: theirs disagree about.
MATCH_TIERS = ("inchikey", "chembl_registry", "inchikey14", "flexmatch")

#: How the two halves were paired, strongest first.
PAIRING_RULES = ("document, same cell line", "document", "compound")

#: Preference between several records of one side inside one document.
SIDE_RULE = ("cell", "censored")


def _key(value: object) -> str | None:
    """A structure key, or None where standardization failed."""
    return value if isinstance(value, str) and value else None


def _side_key(record: dict, cell: str | None) -> tuple:
    return (
        -int(cell is not None and record.get("cell_line") == cell),
        -int(bool(record.get("censored"))),
        str(record.get("source_id")),
    )


def _best(records: list[dict], cell: str | None) -> dict:
    return min(records, key=lambda record: _side_key(record, cell))


class Records:
    """The open-data records, indexed the way the matcher needs them."""

    def __init__(self, curated_open: pd.DataFrame):
        antiviral = antiviral_side(curated_open)
        cytotox = cytotox_side(curated_open)
        self.antiviral_exact = self._index(antiviral, "inchikey")
        self.cytotox_exact = self._index(cytotox, "inchikey")
        self.antiviral = self._index(antiviral, "inchikey14")
        self.cytotox = self._index(cytotox, "inchikey14")

    @staticmethod
    def _index(frame: pd.DataFrame, key: str) -> dict[str, list[dict]]:
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


def _pair(antiviral: list[dict], cytotox: list[dict], cell: str | None) -> dict | None:
    """The organisers' pairing rule: one publication, one record per side."""
    documents = {r["document_id"] for r in antiviral if r.get("document_id")} & {
        r["document_id"] for r in cytotox if r.get("document_id")
    }
    if documents:
        scored = []
        for document in sorted(documents):
            left = _best([r for r in antiviral if r.get("document_id") == document], cell)
            right = _best([r for r in cytotox if r.get("document_id") == document], cell)
            matched = int(left.get("cell_line") == cell) + int(right.get("cell_line") == cell) if cell else 0
            scored.append((-matched, document, left, right))
        _, document, left, right = min(scored, key=lambda item: (item[0], item[1]))
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
            "antiviral": _best(antiviral, cell),
            "cytotox": _best(cytotox, cell),
            "document_id": None,
            "n_documents": 0,
        }
    return None


def recover(split: pd.DataFrame, records: Records, links: pd.DataFrame | None = None) -> pd.DataFrame:
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
            found = _pair(antiviral, cytotox, cell)
            if found is not None:
                chosen = (tier, found)
                break
        row["n_antiviral_records"] = antiviral_seen
        row["n_cytotox_records"] = cytotox_seen
        # No antiviral record and a cytotoxicity record is the shape of the
        # organisers' sentinel: their IC50 column is empty because the
        # compound was inactive, and SI VACV becomes 10000. Flagged, never
        # written into si - vacv.coverage measures how often it is right.
        row["sentinel_evidence"] = bool(cytotox_seen and not antiviral_seen)

        if chosen is not None:
            tier, found = chosen
            left, right = found["antiviral"], found["cytotox"]
            row.update(
                {
                    "match_tier": tier,
                    "rule": found["rule"],
                    "ic50": float(left["value_um"]),
                    "cc50": float(right["value_um"]),
                    "si": float(right["value_um"]) / float(left["value_um"]),
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
    log.info(
        "recover: %d of %d compounds get a pair (%s)",
        int(out["si"].notna().sum()) if "si" in out else 0,
        len(out),
        out["rule"].value_counts().to_dict() if "rule" in out else {},
    )
    return out
