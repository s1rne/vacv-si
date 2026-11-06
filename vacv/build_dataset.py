"""Pair IC50 with CC50 and compute the selectivity index.

Pairing is where a VACV selectivity-index dataset is won or lost. ``SI`` is a
*ratio of two numbers that were meant to be compared*: the authors of a paper
measured antiviral potency and host-cell toxicity in the same cells, with the
same stock solution, read on the same day. Taking a 1986 plaque-reduction IC50
from one laboratory and a 2015 MTT CC50 from another and dividing them produces
a number with a plausible magnitude and no meaning.

So three pairing levels are built, from defensible to merely useful:

``strict``
    Same compound, same publication, same host cell line (both sides known and
    equal). This is the ratio the original authors would have reported.
``document``
    Same compound, same publication, cell line not stated on one side. Most
    older ChEMBL records fall here because the cytotoxicity assay description
    names the cells only in prose.
``compound``
    Same compound, any publication. The antiviral and cytotoxicity halves are
    each aggregated across the whole corpus by median-in-log-space. Highest
    coverage, weakest provenance.

Each compound is emitted once, at the most specific level available for it, and
``pairing_level`` records which. A model that wants only authors' own ratios
filters on ``pairing_level == "strict"``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import classify
from .curate import aggregate
from .paths import KAGGLE, PROCESSED
from .sources.chembl import is_biochemical

log = logging.getLogger(__name__)

PAIRING_LEVELS = ("strict", "document", "compound")

#: Columns the organisers ask for, first and in this order.
REQUIRED_COLUMNS = ["smiles", "ic50", "cc50", "si", "doi"]

PROVENANCE_COLUMNS = [
    "inchikey",
    "inchikey14",
    "pairing_level",
    "ic50_relation",
    "cc50_relation",
    "censored",
    "ic50_type",
    "cc50_type",
    "ic50_n",
    "cc50_n",
    "ic50_log10_spread",
    "cc50_log10_spread",
    "high_variance",
    "cell_line",
    "strain",
    "document_id",
    "source_db",
    "year",
    "mol_weight",
    "n_pairs",
]


def antiviral_side(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows usable as the IC50 half of a VACV selectivity index.

    Two filters beyond the endpoint class. The record must actually name
    vaccinia somewhere - the document widening in the ChEMBL harvest pulls in
    activities against HIV, HSV and influenza from the same papers, and those
    are not VACV potencies. And the assay must not be biochemical: an IC50
    against purified vaccinia VP39 methyltransferase says nothing about the
    concentration that protects a cell.
    """
    mask = frame["measurement_class"].eq("antiviral")
    vaccinia = [
        classify.mentions_vaccinia(row.organism, row.assay_description, row.strain)
        for row in frame.itertuples()
    ]
    mask &= pd.Series(vaccinia, index=frame.index)
    mask &= ~frame["assay_format"].map(is_biochemical).astype(bool)
    return frame[mask].copy()


def cytotox_side(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows usable as the CC50 half.

    No vaccinia filter here, on purpose: cytotoxicity is a property of the
    compound and the host cell, measured on uninfected cells, so a CC50 in Vero
    cells is the same measurement whether the paper went on to test vaccinia or
    herpes. The restriction that matters is the publication, applied by the
    pairing rule.
    """
    mask = frame["measurement_class"].eq("cytotox")
    mask &= ~frame["assay_format"].map(is_biochemical).astype(bool)
    return frame[mask].copy()


def _pair_at_level(antiviral: pd.DataFrame, cytotox: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "strict":
        keys = ["inchikey14", "document_id", "cell_line"]
        left = antiviral[antiviral["cell_line"].notna() & antiviral["document_id"].notna()]
        right = cytotox[cytotox["cell_line"].notna() & cytotox["document_id"].notna()]
    elif level == "document":
        keys = ["inchikey14", "document_id"]
        left = antiviral[antiviral["document_id"].notna()]
        right = cytotox[cytotox["document_id"].notna()]
    elif level == "compound":
        keys = ["inchikey14"]
        left, right = antiviral, cytotox
    else:  # pragma: no cover - guarded by PAIRING_LEVELS
        raise ValueError(level)

    if left.empty or right.empty:
        return pd.DataFrame()

    ic50 = aggregate(left, keys).add_prefix("ic50_")
    cc50 = aggregate(right, keys).add_prefix("cc50_")
    ic50 = ic50.rename(columns={f"ic50_{k}": k for k in keys})
    cc50 = cc50.rename(columns={f"cc50_{k}": k for k in keys})

    paired = ic50.merge(cc50, on=keys, how="inner", suffixes=("_ic", "_cc"))
    if paired.empty:
        return paired
    paired["pairing_level"] = level
    return paired


def _finalise_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    out = paired.copy()
    out["ic50"] = out["ic50_value_um"].astype(float)
    out["cc50"] = out["cc50_value_um"].astype(float)
    out["si"] = out["cc50"] / out["ic50"]
    out["censored"] = ~out["ic50_relation"].isin(["=", "~"]) | ~out["cc50_relation"].isin(["=", "~"])
    out["high_variance"] = out["ic50_high_variance"].astype(bool) | out["cc50_high_variance"].astype(bool)
    out["smiles"] = out["ic50_smiles_std"]
    out["inchikey"] = out["ic50_inchikey"]
    out["mol_weight"] = out["ic50_mol_weight"]
    out["ic50_type"] = out["ic50_standard_type"]
    out["cc50_type"] = out["cc50_standard_type"]
    out["ic50_n"] = out["ic50_n_measurements"]
    out["cc50_n"] = out["cc50_n_measurements"]
    out["doi"] = out.apply(_merge_text(["ic50_doi", "cc50_doi"]), axis=1)
    out["source_db"] = out.apply(_merge_text(["ic50_source_db", "cc50_source_db"]), axis=1)
    out["strain"] = out.apply(_merge_text(["ic50_strain", "cc50_strain"]), axis=1)
    if "cell_line" not in out.columns:
        out["cell_line"] = out.apply(_merge_text(["ic50_cell_line", "cc50_cell_line"]), axis=1)
    if "document_id" not in out.columns:
        out["document_id"] = out.apply(_merge_text(["ic50_document_id", "cc50_document_id"]), axis=1)
    out["year"] = out[["ic50_year", "cc50_year"]].max(axis=1)
    return out


def _merge_text(columns: list[str]):
    def merge(row: pd.Series) -> str | None:
        parts: set[str] = set()
        for column in columns:
            value = row.get(column)
            if isinstance(value, str) and value:
                parts.update(value.split("|"))
        return "|".join(sorted(p for p in parts if p)) or None

    return merge


def _collapse_to_compound(paired: pd.DataFrame) -> pd.DataFrame:
    """One row per compound: median SI in log space across that compound's pairs."""

    def collapse(group: pd.DataFrame) -> pd.Series:
        best = group.iloc[0]
        return pd.Series(
            {
                "smiles": best["smiles"],
                "ic50": float(10 ** np.median(np.log10(group["ic50"].astype(float)))),
                "cc50": float(10 ** np.median(np.log10(group["cc50"].astype(float)))),
                "si": float(10 ** np.median(np.log10(group["si"].astype(float)))),
                "inchikey": best["inchikey"],
                "pairing_level": best["pairing_level"],
                "ic50_relation": "=" if (group["ic50_relation"] == "=").any() else best["ic50_relation"],
                "cc50_relation": "=" if (group["cc50_relation"] == "=").any() else best["cc50_relation"],
                "censored": bool(group["censored"].all()),
                "ic50_type": _join(group["ic50_type"]),
                "cc50_type": _join(group["cc50_type"]),
                "ic50_n": int(group["ic50_n"].sum()),
                "cc50_n": int(group["cc50_n"].sum()),
                "ic50_log10_spread": float(group["ic50_log10_spread"].max()),
                "cc50_log10_spread": float(group["cc50_log10_spread"].max()),
                "high_variance": bool(group["high_variance"].any()),
                "cell_line": _join(group.get("cell_line", pd.Series(dtype=object))),
                "strain": _join(group.get("strain", pd.Series(dtype=object))),
                "document_id": _join(group.get("document_id", pd.Series(dtype=object))),
                "source_db": _join(group["source_db"]),
                "doi": _join(group["doi"]),
                "year": group["year"].max(),
                "mol_weight": float(best["mol_weight"]),
                "n_pairs": len(group),
            }
        )

    return paired.groupby("inchikey14", dropna=False).apply(collapse, include_groups=False).reset_index()


def _join(series: pd.Series) -> str | None:
    parts: set[str] = set()
    for value in series.dropna():
        if isinstance(value, str):
            parts.update(value.split("|"))
    return "|".join(sorted(p for p in parts if p)) or None


def build(curated: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return one table per pairing level plus the merged ``relaxed`` table."""
    antiviral = antiviral_side(curated)
    cytotox = cytotox_side(curated)
    log.info(
        "pairing input: %d antiviral VACV rows (%d compounds), %d cytotox rows (%d compounds)",
        len(antiviral),
        antiviral["inchikey14"].nunique(),
        len(cytotox),
        cytotox["inchikey14"].nunique(),
    )

    tables: dict[str, pd.DataFrame] = {}
    for level in PAIRING_LEVELS:
        paired = _pair_at_level(antiviral, cytotox, level)
        if paired.empty:
            tables[level] = pd.DataFrame(columns=REQUIRED_COLUMNS + PROVENANCE_COLUMNS)
            continue
        tables[level] = _collapse_to_compound(_finalise_pairs(paired))
        log.info("pairing level %s: %d compounds", level, len(tables[level]))

    # Most specific level wins per compound.
    seen: set[str] = set()
    chunks = []
    for level in PAIRING_LEVELS:
        table = tables[level]
        if table.empty:
            continue
        fresh = table[~table["inchikey14"].isin(seen)]
        seen.update(fresh["inchikey14"])
        chunks.append(fresh)
    relaxed = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=REQUIRED_COLUMNS)
    tables["relaxed"] = relaxed
    log.info("relaxed table: %d compounds", len(relaxed))
    return tables


def order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = REQUIRED_COLUMNS + [c for c in PROVENANCE_COLUMNS if c in frame.columns]
    extra = [c for c in frame.columns if c not in columns]
    return frame[[c for c in columns if c in frame.columns] + extra]


def load_competition_set() -> pd.DataFrame | None:
    """Read the organisers' training file from ``data/raw/kaggle/`` if present.

    Absent by design when no Kaggle token exists: the open-source pipeline must
    not depend on it. Column names are guessed, because the competition file is
    not part of this repository and its header is not fixed.
    """
    if not KAGGLE.exists():
        return None
    candidates = sorted(KAGGLE.glob("*.csv"))
    if not candidates:
        return None
    frame = pd.read_csv(candidates[0])
    lowered = {c.lower().strip(): c for c in frame.columns}
    smiles_column = next((lowered[k] for k in lowered if "smiles" in k), None)
    if smiles_column is None:
        log.warning("competition file %s has no SMILES column", candidates[0])
        return None
    renames = {smiles_column: "smiles"}
    for target in ("ic50", "cc50", "si"):
        match = next((lowered[k] for k in lowered if k.replace("_", "").startswith(target)), None)
        if match:
            renames[match] = target
    out = frame.rename(columns=renames)
    out["source_file"] = candidates[0].name
    return out


def write(tables: dict[str, pd.DataFrame], directory: Path = PROCESSED) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    files = {
        "vacv_si.csv": tables["relaxed"],
        "vacv_si_strict.csv": tables["strict"],
        "vacv_si_relaxed.csv": tables["relaxed"],
    }
    for name, table in files.items():
        path = directory / name
        order_columns(table).to_csv(path, index=False)
        written[name] = path
        log.info("wrote %s (%d rows)", path, len(table))
    return written
