"""Checks that must pass before a dataset is handed to a model or to judges.

Each check returns a :class:`Finding`. ``severity="error"`` means the file is
not fit to train on; ``"warning"`` means a real caveat that is documented
rather than fixed. :func:`report` prints both and :func:`assert_ok` raises on
any error, so the same code backs ``uv run python -m vacv.cli validate`` and the
pytest suite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .build_dataset import REQUIRED_COLUMNS
from .units import MAX_UM, MIN_UM

RDLogger.DisableLog("rdApp.*")

#: Relative tolerance for si == cc50 / ic50. The three columns are written as
#: decimal text, so only rounding may differ.
SI_TOLERANCE = 1e-9


@dataclass
class Finding:
    check: str
    severity: str  # error | warning | info
    message: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.upper():7}] {self.check}: {self.message}"


def check_schema(frame: pd.DataFrame) -> Iterable[Finding]:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        yield Finding("schema", "error", f"missing required columns: {missing}")
        return
    if list(frame.columns[: len(REQUIRED_COLUMNS)]) != REQUIRED_COLUMNS:
        yield Finding(
            "schema",
            "warning",
            "required columns are present but not first; organisers asked for "
            f"{REQUIRED_COLUMNS}",
        )
    for column in ("ic50", "cc50", "si"):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            yield Finding("schema", "error", f"{column} is not numeric")
    yield Finding("schema", "info", f"{len(frame)} rows, {len(frame.columns)} columns")


def check_smiles(frame: pd.DataFrame) -> Iterable[Finding]:
    bad = [s for s in frame["smiles"] if not isinstance(s, str) or Chem.MolFromSmiles(s) is None]
    if bad:
        yield Finding("smiles", "error", f"{len(bad)} SMILES do not parse", {"examples": bad[:5]})
    else:
        yield Finding("smiles", "info", f"all {len(frame)} SMILES parse")

    multi = [s for s in frame["smiles"] if isinstance(s, str) and "." in s]
    if multi:
        yield Finding("smiles", "error", f"{len(multi)} SMILES still contain a mixture", {"examples": multi[:5]})


def check_units(frame: pd.DataFrame) -> Iterable[Finding]:
    for column in ("ic50", "cc50"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            yield Finding("units", "error", f"{int(values.isna().sum())} non-numeric {column} values")
        outside = values[(values < MIN_UM) | (values > MAX_UM)]
        if len(outside):
            yield Finding(
                "units",
                "error",
                f"{len(outside)} {column} values outside [{MIN_UM}, {MAX_UM}] uM",
                {"min": float(values.min()), "max": float(values.max())},
            )
        else:
            yield Finding(
                "units",
                "info",
                f"{column} spans {values.min():.4g} to {values.max():.4g} uM "
                f"(median {values.median():.4g})",
            )


def check_duplicates(frame: pd.DataFrame) -> Iterable[Finding]:
    key = "inchikey14" if "inchikey14" in frame.columns else "smiles"
    duplicated = frame[key][frame[key].duplicated(keep=False)]
    if len(duplicated):
        yield Finding(
            "duplicates",
            "error",
            f"{duplicated.nunique()} {key} values appear more than once",
            {"examples": sorted(set(duplicated))[:5]},
        )
    else:
        yield Finding("duplicates", "info", f"{key} is unique across {len(frame)} rows")


def check_si_consistency(frame: pd.DataFrame) -> Iterable[Finding]:
    expected = frame["cc50"].astype(float) / frame["ic50"].astype(float)
    actual = frame["si"].astype(float)
    relative = (actual - expected).abs() / expected.abs()
    offenders = frame[relative > SI_TOLERANCE]
    if len(offenders):
        yield Finding(
            "si_consistency",
            "error",
            f"{len(offenders)} rows where si != cc50/ic50",
            {"max_relative_error": float(relative.max())},
        )
    else:
        yield Finding("si_consistency", "info", "si == cc50/ic50 for every row")

    non_positive = frame[(frame["si"] <= 0) | ~np.isfinite(frame["si"].astype(float))]
    if len(non_positive):
        yield Finding("si_consistency", "error", f"{len(non_positive)} rows with non-positive or non-finite si")


def check_provenance(frame: pd.DataFrame) -> Iterable[Finding]:
    with_doi = frame["doi"].notna() & frame["doi"].astype(str).str.strip().ne("")
    share = float(with_doi.mean()) if len(frame) else math.nan
    yield Finding(
        "provenance",
        "info",
        f"{int(with_doi.sum())} of {len(frame)} rows carry a DOI ({share:.1%})",
    )
    if share < 0.5:
        yield Finding("provenance", "warning", f"only {share:.1%} of rows carry a DOI")

    if "pairing_level" in frame.columns:
        counts = frame["pairing_level"].value_counts().to_dict()
        yield Finding("provenance", "info", f"pairing levels: {counts}")
        weak = counts.get("compound", 0)
        if weak and weak / len(frame) > 0.5:
            yield Finding(
                "provenance",
                "warning",
                f"{weak / len(frame):.1%} of rows pair IC50 and CC50 across different "
                "publications; see docs/dataset.md",
            )

    if "censored" in frame.columns:
        censored = frame["censored"].astype(bool).sum()
        yield Finding("provenance", "info", f"{int(censored)} rows involve a censored (> or <) measurement")

    if "high_variance" in frame.columns:
        noisy = frame["high_variance"].astype(bool).sum()
        yield Finding(
            "provenance",
            "info",
            f"{int(noisy)} rows aggregate replicate measurements spanning more than one log unit",
        )


CHECKS = (
    check_schema,
    check_smiles,
    check_units,
    check_duplicates,
    check_si_consistency,
    check_provenance,
)


def validate(frame: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = list(check_schema(frame))
    if errors(findings):
        # Every later check indexes the required columns by name; running them
        # on a frame that is missing one only produces a KeyError traceback
        # instead of the finding that already explains the problem.
        return findings
    for check in CHECKS[1:]:
        findings.extend(check(frame))
    return findings


def report(frame: pd.DataFrame) -> list[Finding]:
    findings = validate(frame)
    for finding in findings:
        print(finding)
        if finding.details:
            print(f"          {finding.details}")
    return findings


def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]


def assert_ok(frame: pd.DataFrame) -> None:
    failed = errors(validate(frame))
    if failed:
        raise AssertionError("; ".join(str(f) for f in failed))
