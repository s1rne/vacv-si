"""The organisers' own training table, when it is available locally.

Not an open database and not downloaded by this pipeline: no Kaggle credentials
are assumed, and the competition rules govern the file, not an open licence. The
pipeline is built so that every number in ``data/processed/`` comes from public
databases, and this module only lets the competition file be folded in as an
extra source when somebody drops it into ``data/raw/kaggle/``.

Expected: any CSV with a SMILES column and some of IC50 / CC50 / SI. Column
names are matched case-insensitively because the file is not part of this
repository and its header is not under our control.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..paths import KAGGLE
from ..schema import conform

log = logging.getLogger(__name__)

SOURCE = "kaggle"

#: Assumed unit of the organisers' IC50/CC50 columns. The competition data card
#: states micromolar; if that ever changes, this is the one line to edit.
ASSUMED_UNITS = "uM"


def available() -> bool:
    return KAGGLE.exists() and any(KAGGLE.glob("*.csv"))


def _column(frame: pd.DataFrame, *candidates: str) -> str | None:
    lowered = {c.lower().replace(" ", "").replace("_", ""): c for c in frame.columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in lowered:
            return lowered[key]
    for key, original in lowered.items():
        if any(key.startswith(c.lower()) for c in candidates):
            return original
    return None


def harvest() -> pd.DataFrame:
    if not available():
        log.info("kaggle: no competition CSV under %s, skipping", KAGGLE)
        return conform(pd.DataFrame())

    rows: list[dict] = []
    for path in sorted(KAGGLE.glob("*.csv")):
        frame = pd.read_csv(path)
        smiles_column = _column(frame, "smiles", "canonical_smiles", "structure")
        if smiles_column is None:
            log.warning("kaggle: %s has no SMILES column", path.name)
            continue
        ic50_column = _column(frame, "ic50")
        cc50_column = _column(frame, "cc50")
        for index, record in frame.iterrows():
            for column, endpoint, measurement_class in (
                (ic50_column, "IC50", "antiviral"),
                (cc50_column, "CC50", "cytotox"),
            ):
                if column is None or pd.isna(record[column]):
                    continue
                rows.append(
                    {
                        "smiles": record[smiles_column],
                        "standard_type": endpoint,
                        "measurement_class": measurement_class,
                        "value_um": None,
                        "relation": "=",
                        "raw_value": record[column],
                        "raw_units": ASSUMED_UNITS,
                        "assay_description": f"competition training set ({path.name})",
                        "assay_format": None,
                        "organism": "Vaccinia virus" if measurement_class == "antiviral" else None,
                        "strain": "VACV",
                        "cell_line": None,
                        "source_db": "Kaggle competition",
                        "source_id": f"KAGGLE_{path.stem}_{index}_{endpoint}",
                        "assay_id": f"KAGGLE_{path.stem}",
                        "document_id": f"KAGGLE_{path.stem}",
                        "doi": None,
                        "doi_provenance": None,
                        "reference": f"MEPhI VACV SI competition, {path.name}",
                        "year": None,
                        "activity_comment": None,
                        "data_validity": None,
                    }
                )
        log.info("kaggle: %s -> %d measurements", path.name, len(rows))

    frame = conform(pd.DataFrame(rows))
    frame["value_um"] = pd.to_numeric(frame["raw_value"], errors="coerce")
    return frame
