"""Decide what a measurement actually measures.

Public antiviral data is free text. The same number can arrive as
``standard_type="IC50"`` with the description "inhibition of vaccinia virus
replication in Vero cells" (an antiviral potency) or as ``standard_type="IC50"``
with "cytotoxicity against uninfected Vero cells" (the toxicity half of the
pair). Pairing those two the wrong way round inverts the selectivity index, so
the rules below are explicit, ordered, and tested.
"""

from __future__ import annotations

import re

#: Endpoint names that are unambiguously cytotoxicity / tolerated dose.
CYTOTOX_TYPES = {
    "CC50",
    "TC50",
    "CD50",
    "TD50",
    "GI50",
    "TGI",
    "MNTD",
    "MTC",
    "MTD",
    # "Minimum cytotoxic concentration": the lowest concentration that alters
    # cell morphology. It is how the 1980s and 1990s antiviral literature -
    # the De Clercq series in PRK, HEL and E6SM cells, which is where a large
    # part of the competition set comes from - reports the cytotoxic side.
    "MCC",
    "LC50",
    "LD50",
    "CT50",
}

#: Endpoint names for a half-maximal effect whose meaning depends on the assay.
AMBIGUOUS_TYPES = {
    "IC50",
    "EC50",
    "ED50",
    "AC50",
    "Potency",
    "IC50ratio",
    "I50",
    "Activity",
    # In the antiviral literature "minimum inhibitory concentration" is not
    # the antibacterial MIC: the assay description spells out "required to
    # reduce vaccinia virus induced cytopathogenicity by 50% in primary
    # rabbit kidney cells", which is an EC50 under another name. It stays in
    # the ambiguous set so the free text, not the label, decides - a genuine
    # antibacterial MIC in the same paper says nothing about a virus and ends
    # up unclassified, as it should.
    "MIC",
    "MIC50",
    "ID50",
}

#: Endpoint names under which a paper reports the selectivity index it
#: computed itself. These are not a half of a pair, they are a (rounded)
#: answer: the authors' own CC50/EC50. Useful where no pair can be rebuilt,
#: and no substitute for one where it can - the organisers recomputed the
#: ratio from the two columns, so a paper's printed "71" sits next to their
#: "71.391".
SELECTIVITY_TYPES = {
    "Ratio CC50/EC50",
    "Ratio CC50/IC50",
    "Selectivity Index",
    "Selectivity index",
    "SI",
    "Therapeutic index",
    "Therapeutic Index",
    "TI",
}

#: Endpoint names that cannot take part in an SI pair.
IGNORED_TYPES = {
    "Ki",
    "Kd",
    "Km",
    "Kb",
    "IC90",
    "EC90",
    "IC95",
    "EC99",
    "MIC90",
    "Inhibition",
    "Activity at",
    "T1/2",
    "CL",
    "Papp",
    "Solubility",
    "logP",
    "logD",
}

_CYTOTOX_TEXT = re.compile(
    r"cytotox|cytopath(?:ic)?\s+(?:effect\s+)?(?:on|in)\s+uninfected"
    r"|cell\s+viabilit|cell\s+growth|growth\s+inhibition"
    r"|uninfected|mock[- ]infect|non[- ]infect|host\s+cell\s+toxicit"
    r"|toxicit|tolerated|cell\s+proliferation|antiproliferative"
    r"|minimum\s+(?:cytotoxic|inhibitory\s+concentration\s+for\s+cell)",
    re.I,
)

_ANTIVIRAL_TEXT = re.compile(
    r"antiviral|anti[- ]viral|virus\s+(?:replication|multiplication|yield|growth|titer|titre)"
    r"|plaque\s+(?:reduction|formation|number)|cytopathic\s+effect|cytopathogenic"
    r"|\bcpe\b|virus[- ]induced|viral\s+(?:replication|inhibition|infection|load)"
    r"|inhibition\s+of\s+(?:the\s+)?(?:virus|viral|vaccinia|vacv)"
    r"|against\s+(?:vaccinia|vacv|poxvirus|cowpox|variola|monkeypox)",
    re.I,
)

_VACCINIA_TEXT = re.compile(
    r"vaccinia|\bvacv\b|\bvv\b(?!\w)|orthopoxvirus\s+vaccinia|modified\s+vaccinia", re.I
)

#: VACV strains seen in the literature, longest pattern first so that
#: "Western Reserve" wins over a bare "WR".
_STRAINS: list[tuple[str, re.Pattern[str]]] = [
    ("WR", re.compile(r"western\s+reserve|\bwr\b|strain\s+wr", re.I)),
    ("Copenhagen", re.compile(r"copenhagen", re.I)),
    ("IHD-J", re.compile(r"ihd[-\s]?j", re.I)),
    ("IHD", re.compile(r"\bihd\b", re.I)),
    ("Lister", re.compile(r"\blister\b|elstree", re.I)),
    ("Ankara", re.compile(r"ankara|\bmva\b", re.I)),
    ("NYCBH", re.compile(r"nycbh|new\s+york\s+city\s+board", re.I)),
    ("Tian Tan", re.compile(r"tian\s*tan", re.I)),
    ("Wyeth", re.compile(r"\bwyeth\b", re.I)),
]

#: Host cell lines used in poxvirus work. Order matters: BSC-40 before BSC-1.
_CELL_LINES: list[tuple[str, re.Pattern[str]]] = [
    ("BSC-40", re.compile(r"bsc[-\s]?40", re.I)),
    ("BSC-1", re.compile(r"bsc[-\s]?1\b|bs[-\s]?c[-\s]?1", re.I)),
    ("Vero", re.compile(r"\bvero\b(?![-\s]?e6)|vero[-\s]?76|vero[-\s]?e6", re.I)),
    ("HeLa", re.compile(r"\bhela\b", re.I)),
    ("CV-1", re.compile(r"\bcv[-\s]?1\b", re.I)),
    ("HFF", re.compile(r"\bhff\b|human\s+foreskin\s+fibroblast", re.I)),
    ("MRC-5", re.compile(r"\bmrc[-\s]?5\b", re.I)),
    ("HEL", re.compile(r"\bhel\b|human\s+embryonic\s+lung", re.I)),
    ("HEp-2", re.compile(r"\bhep[-\s]?2\b", re.I)),
    ("A549", re.compile(r"\ba549\b", re.I)),
    ("L929", re.compile(r"\bl[-\s]?929\b", re.I)),
    ("RK-13", re.compile(r"\brk[-\s]?13\b", re.I)),
    ("LLC-MK2", re.compile(r"llc[-\s]?mk2", re.I)),
    ("MDCK", re.compile(r"\bmdck\b", re.I)),
    ("3T3", re.compile(r"\b3t3\b|nih[-\s]?3t3", re.I)),
    ("PRK", re.compile(r"primary\s+rabbit\s+kidney|\bprk\b", re.I)),
    ("CEF", re.compile(r"chick(?:en)?\s+embryo\s+fibroblast|\bcef\b", re.I)),
    ("KB", re.compile(r"\bkb\s+cell", re.I)),
    ("MT-4", re.compile(r"\bmt[-\s]?4\b", re.I)),
]


def _text(*parts: object) -> str:
    """Join only the parts that are real strings.

    Source frames arrive from pandas, where a missing string is ``float('nan')``
    rather than ``None``, and a naive truthiness filter lets it through.
    """
    return " ".join(part for part in parts if isinstance(part, str) and part)


def normalise_type(standard_type: str | None) -> str | None:
    if standard_type is None:
        return None
    text = str(standard_type).strip()
    return text or None


def _normalised(names: set[str]) -> set[str]:
    return {name.replace(" ", "").upper() for name in names}


def classify(
    standard_type: str | None,
    *,
    description: str | None = None,
    organism: str | None = None,
    assay_type: str | None = None,
) -> str:
    """Return ``"antiviral"``, ``"cytotox"``, ``"selectivity"`` or ``"unknown"``.

    ``assay_type`` is the ChEMBL assay type letter (``T`` toxicity, ``A``
    ADMET, ``F`` functional, ``B`` binding) and is only consulted when the
    free text is silent.
    """
    endpoint = normalise_type(standard_type)
    if endpoint is None:
        return "unknown"
    key = endpoint.replace(" ", "").upper()
    if key in _normalised(SELECTIVITY_TYPES):
        return "selectivity"
    if key in _normalised(CYTOTOX_TYPES):
        return "cytotox"
    if key not in _normalised(AMBIGUOUS_TYPES):
        return "unknown"

    text = _text(description, organism)
    cytotox_hit = bool(_CYTOTOX_TEXT.search(text))
    antiviral_hit = bool(_ANTIVIRAL_TEXT.search(text))

    # "cytotoxicity measured as inhibition of virus-induced CPE" does happen;
    # when both fire, the antiviral wording is the specific claim only if the
    # cytotoxic wording is not attached to uninfected cells.
    if cytotox_hit and not antiviral_hit:
        return "cytotox"
    if antiviral_hit and not cytotox_hit:
        return "antiviral"
    if cytotox_hit and antiviral_hit:
        if re.search(r"uninfected|mock[- ]infect|non[- ]infect", text, re.I):
            return "cytotox"
        return "antiviral"
    if (assay_type or "").upper() in {"T", "A"}:
        return "cytotox"
    return "unknown"


def mentions_antiviral(*texts: object) -> bool:
    """Does this text describe an antiviral measurement at all?

    Needed where the endpoint name carries no information - a record whose
    type is "Inhibition" and whose value is the words "Not active" cannot be
    classified from its label, and the label is not what matters about it.
    """
    return bool(_ANTIVIRAL_TEXT.search(_text(*texts)))


def mentions_vaccinia(*texts: object) -> bool:
    return bool(_VACCINIA_TEXT.search(_text(*texts)))


def extract_strain(*texts: object) -> str | None:
    joined = _text(*texts)
    if not joined:
        return None
    for name, pattern in _STRAINS:
        if pattern.search(joined):
            return name
    return None


def extract_cell_line(*texts: object) -> str | None:
    joined = _text(*texts)
    if not joined:
        return None
    for name, pattern in _CELL_LINES:
        if pattern.search(joined):
            return name
    return None
