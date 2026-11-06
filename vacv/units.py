"""Conversion of reported concentrations to micromolar.

Getting this wrong is the single cheapest way to poison the model: a mass
concentration left untouched is off by roughly the molecular weight, i.e. two
to three orders of magnitude, and it is off *in the same direction for IC50 and
CC50*, so the selectivity index looks plausible while both halves are wrong.
Anything not convertible returns ``None`` and is dropped with a counted reason
rather than guessed.
"""

from __future__ import annotations

import math
import re

#: Multiplicative factors to micromolar for molar units.
MOLAR = {
    "m": 1e6,
    "mol/l": 1e6,
    "mol l-1": 1e6,
    "mm": 1e3,
    "mmol/l": 1e3,
    "um": 1.0,
    "umol/l": 1.0,
    "nm": 1e-3,
    "nmol/l": 1e-3,
    "pm": 1e-6,
    "pmol/l": 1e-6,
    "fm": 1e-9,
}

#: Multiplicative factors to micrograms per millilitre for mass units.
MASS_UG_PER_ML = {
    "g/l": 1e3,
    "mg/ml": 1e3,
    "mg/l": 1.0,
    "ug/ml": 1.0,
    "mcg/ml": 1.0,
    "ug/l": 1e-3,
    "ng/ml": 1e-3,
    "ng/l": 1e-6,
    "pg/ml": 1e-6,
}

#: Units that encode a negative decadic logarithm of a molar concentration.
LOG_MOLAR = {"log m", "-log m", "log10 m", "-log10 m", "logm", "pm-log"}

_SUPERSCRIPT = str.maketrans({"µ": "u", "μ": "u"})


def canonical_units(units: str | None) -> str | None:
    """Reduce the many spellings used by public databases to one token."""
    if units is None:
        return None
    text = str(units).strip().lower().translate(_SUPERSCRIPT)
    if not text or text in {"none", "nan", "unspecified"}:
        return None
    text = text.replace("micro", "u").replace("Â", "")
    text = re.sub(r"\s+", " ", text)
    # ChEMBL writes "ug.mL-1", "10'-6M", "nmol.L-1"; PubChem writes "uM".
    text = text.replace("'", "").replace(".", " ")
    text = re.sub(r"\b(ml|l|dl)\s*-1\b", r"/\1", text)
    text = re.sub(r"\b(mol)\s*-1\b", r"\1", text)
    text = text.replace(" /", "/").replace("/ ", "/")
    text = text.replace("litre", "l").replace("liter", "l")
    text = text.replace("molar", "m")
    text = text.replace(" ", "")
    aliases = {
        "umol/l": "umol/l",
        "nmol/l": "nmol/l",
        "mmol/l": "mmol/l",
        "mol/l": "mol/l",
        "ugml-1": "ug/ml",
        "mgml-1": "mg/ml",
        "ugml": "ug/ml",
        "mgl-1": "mg/l",
        "10-6m": "um",
        "10-9m": "nm",
        "10-3m": "mm",
        "-logm": "-log m",
        "logm": "log m",
    }
    return aliases.get(text, text)


def to_micromolar(
    value: float | str | None,
    units: str | None,
    *,
    mol_weight: float | None = None,
) -> tuple[float | None, str]:
    """Return ``(micromolar, reason)``; ``micromolar`` is ``None`` on failure.

    ``reason`` is ``"ok"`` on success and otherwise names the obstacle, so the
    caller can report *why* records were lost instead of silently shrinking.
    """
    if value is None:
        return None, "no_value"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "unparsable_value"
    if not math.isfinite(number):
        return None, "non_finite_value"

    token = canonical_units(units)
    if token is None:
        return None, "no_units"
    if token in MOLAR:
        result = number * MOLAR[token]
    elif token in MASS_UG_PER_ML:
        if not mol_weight or mol_weight <= 0:
            return None, "mass_units_without_mw"
        result = number * MASS_UG_PER_ML[token] * 1000.0 / mol_weight
    elif token in LOG_MOLAR or token.replace("-", "") in LOG_MOLAR:
        # A reported -log10(M) of 6 is 1 uM.
        result = 10.0 ** (6.0 - number)
    else:
        return None, f"unsupported_units:{token}"

    if not math.isfinite(result) or result <= 0:
        return None, "non_positive_result"
    return result, "ok"


def from_pchembl(pchembl: float | str | None) -> float | None:
    """ChEMBL's pChEMBL value is -log10(molar); 6.0 means 1 uM."""
    if pchembl in (None, ""):
        return None
    try:
        number = float(pchembl)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return 10.0 ** (6.0 - number)


def canonical_relation(relation: str | None) -> str:
    if relation is None:
        return "="
    text = str(relation).strip()
    mapping = {
        "": "=",
        "=": "=",
        "==": "=",
        "~": "~",
        "> ": ">",
        ">": ">",
        ">=": ">=",
        ">>": ">",
        "<": "<",
        "<=": "<=",
        "<<": "<",
    }
    return mapping.get(text, "=")


#: Plausibility window for a cell-based potency or cytotoxicity measurement.
#: 1 pM is below anything a CPE or plaque assay resolves; 10 mM is above the
#: solubility of almost any screened compound in culture medium.
MIN_UM = 1e-6
MAX_UM = 1e4


def plausible(value_um: float | None) -> bool:
    return value_um is not None and MIN_UM <= value_um <= MAX_UM
