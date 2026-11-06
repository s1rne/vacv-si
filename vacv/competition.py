"""The competition compounds as structures and labels, nothing else.

Only ``SMILES``, ``Title``, ``strain`` and ``Culture`` are read here.
``CC50_uM``, ``IC50_uM_VACV`` and ``SI VACV`` are never touched: this module
exists so the targeted harvesters know *what to look up in the open
databases*, and handing them the organisers' own measurements would defeat the
point of looking anything up.

Several ``Title`` values are database identifiers rather than a paper's
compound label - 378 of the 917 rows are PubChem SIDs (``sid103460068``) and
four are ChEMBL IDs (``CHEMBL6102784``). Those resolve to a database record
directly, without going through a structure match, which is what
:mod:`vacv.sources.identifiers` does with them.
"""

from __future__ import annotations

import re

import pandas as pd

from .curate import standardize
from .paths import KAGGLE

SPLITS = ("train", "test")

_SID = re.compile(r"^sid\s*(\d+)$", re.I)
_CID = re.compile(r"^cid\s*(\d+)$", re.I)
_CHEMBL = re.compile(r"^(CHEMBL\d+)$", re.I)


def load_split(name: str) -> pd.DataFrame:
    """One competition split with standardized structure keys attached.

    ``inchikey`` is the exact standardized parent, ``inchikey14`` its skeleton
    (connectivity only, so stereochemistry and protonation are ignored) and
    ``inchikey14_untautomerized`` the same skeleton computed without the
    canonical-tautomer step. The last one exists because the tautomer
    canonicaliser is the one part of standardization that can land two
    spellings of the same molecule on two different keys.
    """
    frame = pd.read_csv(KAGGLE / f"{name}.csv")
    standardized = [standardize(s) for s in frame["SMILES"]]
    plain = [standardize(s, canonical_tautomer=False) for s in frame["SMILES"]]
    frame["inchikey"] = [s.inchikey for s in standardized]
    frame["inchikey14"] = [s.inchikey[:14] if s.inchikey else None for s in standardized]
    frame["inchikey14_untautomerized"] = [s.inchikey[:14] if s.inchikey else None for s in plain]
    frame["smiles_std"] = [s.smiles for s in standardized]
    frame["standardize_reject"] = [s.reject for s in standardized]
    frame["split"] = name
    return frame


def splits() -> dict[str, pd.DataFrame]:
    return {name: load_split(name) for name in SPLITS if (KAGGLE / f"{name}.csv").exists()}


def all_compounds() -> pd.DataFrame:
    """Train and test in one frame, or an empty frame when neither is present."""
    found = splits()
    if not found:
        return pd.DataFrame(columns=["split", "Title", "SMILES", "inchikey", "inchikey14"])
    return pd.concat(found.values(), ignore_index=True)


def title_identifier(title: object) -> tuple[str, str] | None:
    """``("sid", "103460068")``, ``("cid", …)``, ``("chembl", "CHEMBL123")`` or None."""
    if not isinstance(title, str):
        return None
    text = title.strip()
    for kind, pattern in (("sid", _SID), ("cid", _CID), ("chembl", _CHEMBL)):
        found = pattern.match(text)
        if found:
            return kind, found.group(1).upper()
    return None


def identifiers(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-compound database identifier taken from ``Title``, where there is one."""
    frame = all_compounds() if frame is None else frame
    rows = []
    for record in frame.to_dict("records"):
        found = title_identifier(record.get("Title"))
        if found is None:
            continue
        kind, value = found
        rows.append(
            {
                "split": record.get("split"),
                "Title": record.get("Title"),
                "kind": kind,
                "value": value,
                "inchikey": record.get("inchikey"),
                "inchikey14": record.get("inchikey14"),
            }
        )
    return pd.DataFrame(rows, columns=["split", "Title", "kind", "value", "inchikey", "inchikey14"])
