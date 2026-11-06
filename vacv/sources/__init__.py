"""Open-source harvesters.

Each module exposes ``harvest() -> pandas.DataFrame`` conforming to
:data:`vacv.schema.SOURCE_COLUMNS`. One row is one measurement.

Sources that were evaluated and *not* implemented (BindingDB bulk download,
PDBbind, DrugBank, the Broad Drug Repurposing Hub, NCATS OpenData) are listed
with the evidence in ``docs/data_sources.md``. The short version: for
cell-based vaccinia potency paired with host-cell cytotoxicity, ChEMBL and
PubChem BioAssay are the whole of the open world, and PubChem is 98% ChEMBL
re-deposits.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from . import chembl, chembl_targeted, kaggle_competition, pubchem

#: Harvesters that query open databases. ``kaggle`` is deliberately absent:
#: it reads a local competition file, is not an open source, and must never be
#: required for the pipeline to run.
#:
#: ``chembl_targeted`` is a borderline case and is in: it queries ChEMBL only,
#: but it decides *what* to query from the competition compounds' SMILES and
#: Titles. With no competition file it returns an empty frame instead of
#: failing, so the dataset still builds from the open world alone.
OPEN_SOURCES: dict[str, Callable[[], pd.DataFrame]] = {
    "chembl": chembl.harvest,
    "chembl_targeted": chembl_targeted.harvest,
    "pubchem": pubchem.harvest,
}

ALL_SOURCES: dict[str, Callable[[], pd.DataFrame]] = {
    **OPEN_SOURCES,
    "kaggle": kaggle_competition.harvest,
}

__all__ = ["OPEN_SOURCES", "ALL_SOURCES", "chembl", "pubchem", "kaggle_competition"]
