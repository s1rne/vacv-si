"""Filesystem layout shared by every stage of the pipeline."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("VACV_ROOT", Path(__file__).resolve().parent.parent))

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
KAGGLE = RAW / "kaggle"

#: Competition compound -> ChEMBL molecule links, written by the targeted
#: harvest and read by the matcher.
MOLECULE_LINKS = INTERIM / "competition_molecules.csv"

#: Records that report inactivity against vaccinia without a number, written
#: by the curate stage and read by the matcher.
INACTIVE = INTERIM / "inactive.csv.gz"
DOCS = ROOT / "docs"


def raw_dir(source: str) -> Path:
    """Cache directory for one source's untouched API responses."""
    path = RAW / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs() -> None:
    for path in (RAW, INTERIM, PROCESSED, DOCS):
        path.mkdir(parents=True, exist_ok=True)
