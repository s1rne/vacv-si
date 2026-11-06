"""The normalized record every source module must return.

One row is one measurement, not one compound. Pairing IC50 with CC50 happens
later in :mod:`vacv.build_dataset`, so a source module must never invent the
missing half of a pair.
"""

from __future__ import annotations

import pandas as pd

#: Columns of the per-measurement source table, in order.
SOURCE_COLUMNS: list[str] = [
    "smiles",  # SMILES exactly as the source reports it
    "standard_type",  # IC50 / EC50 / ED50 / CC50 / TC50 / AC50 / ...
    "measurement_class",  # antiviral | cytotox | unknown (see vacv.classify)
    "value_um",  # measurement converted to micromolar, NaN if impossible
    "relation",  # =, >, <, >=, <=, ~
    "raw_value",  # value as reported
    "raw_units",  # units as reported
    "assay_description",
    "assay_format",  # BAO label: organism-based / cell-based / single protein / ...
    "organism",  # assay / target organism string
    "strain",  # VACV strain when stated (WR, Copenhagen, IHD-J, Lister, ...)
    "cell_line",  # host cell line when stated (Vero, HeLa, BSC-1, HFF, ...)
    "source_db",  # ChEMBL / PubChem / Kaggle / ...
    "source_id",  # stable per-measurement identifier inside the source
    "assay_id",
    "document_id",  # publication grouping key, used by the strict pairing rule
    "doi",
    "doi_provenance",  # document | assay_citation - see docs/data_sources.md
    "reference",  # human readable citation
    "year",
    "activity_comment",  # free text caveats the source attaches
    "data_validity",  # source-side data quality flag
]

def conform(frame: pd.DataFrame) -> pd.DataFrame:
    """Add missing columns, drop unknown ones, fix the order."""
    out = frame.copy()
    for column in SOURCE_COLUMNS:
        if column not in out.columns:
            out[column] = None
    out = out[SOURCE_COLUMNS]
    out["value_um"] = pd.to_numeric(out["value_um"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    for column in ("smiles", "standard_type", "relation", "source_db", "source_id"):
        out[column] = out[column].astype("object")
    return out.reset_index(drop=True)
