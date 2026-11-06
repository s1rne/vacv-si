"""How much of the competition set can be rebuilt from open databases?

The competition compounds were taken from published papers on anti-vaccinia
activity, so the same molecules should reappear in ChEMBL with the same
numbers. This module measures that overlap, and - more usefully - measures how
well a selectivity index reconstructed purely from open sources agrees with the
organisers' own value on the training compounds, where the answer is known.

That second number is the honest estimate of how much signal an externally
harvested label carries. It is reported as a log10 error, because an SI that is
right to within a factor of two is useful and an SI that is wrong by a factor
of a thousand is not, and a linear error cannot tell those apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .build_dataset import antiviral_side, build, cytotox_side
from .curate import standardize
from .paths import KAGGLE

log = logging.getLogger(__name__)

#: The organisers use 10000 as a sentinel for "selectivity could not be
#: determined", not as a measured ratio: it appears on rows whose IC50 is
#: missing because the compound was inactive at every tested dose.
SI_SENTINEL = 10000.0


@dataclass
class Overlap:
    split: str
    compounds: int
    parsed: int
    matched_any: int
    matched_antiviral: int
    matched_cytotox: int
    matched_both: int
    paired_si: int

    def as_row(self) -> list:
        return [
            self.split,
            self.compounds,
            self.parsed,
            self.matched_any,
            self.matched_antiviral,
            self.matched_cytotox,
            self.matched_both,
            self.paired_si,
        ]


HEADER = [
    "split",
    "rows",
    "standardized",
    "found in open data",
    "with VACV IC50/EC50",
    "with CC50",
    "with both",
    "SI reconstructable",
]


def load_split(name: str) -> pd.DataFrame:
    path = KAGGLE / f"{name}.csv"
    frame = pd.read_csv(path)
    standardized = [standardize(s) for s in frame["SMILES"]]
    frame["inchikey"] = [s.inchikey for s in standardized]
    frame["inchikey14"] = [s.inchikey[:14] if s.inchikey else None for s in standardized]
    frame["standardize_reject"] = [s.reject for s in standardized]
    frame["split"] = name
    return frame


def open_source_rows(curated: pd.DataFrame) -> pd.DataFrame:
    """Curated measurements with the competition's own rows removed."""
    return curated[curated["source_db"].astype(str) != "Kaggle competition"].copy()


def overlap(split: pd.DataFrame, curated_open: pd.DataFrame, paired: pd.DataFrame, key: str) -> Overlap:
    antiviral = set(antiviral_side(curated_open)[key].dropna())
    cytotox = set(cytotox_side(curated_open)[key].dropna())
    everything = set(curated_open[key].dropna())
    keys = split[key].dropna()
    return Overlap(
        split=f"{split['split'].iat[0]} ({key})",
        compounds=len(split),
        parsed=int(split["inchikey"].notna().sum()),
        matched_any=int(keys.isin(everything).sum()),
        matched_antiviral=int(keys.isin(antiviral).sum()),
        matched_cytotox=int(keys.isin(cytotox).sum()),
        matched_both=int(keys.isin(antiviral & cytotox).sum()),
        paired_si=int(keys.isin(set(paired[key].dropna())).sum()) if key in paired.columns else 0,
    )


def agreement(train: pd.DataFrame, paired: pd.DataFrame, key: str = "inchikey14") -> pd.DataFrame:
    """Join the organisers' SI to the open-data SI for the same molecule."""
    left = train[[key, "Title", "SI VACV", "CC50_uM", "IC50_uM_VACV", "Culture"]].dropna(subset=[key])
    right = paired[[key, "si", "ic50", "cc50", "pairing_level", "doi", "censored"]].rename(
        columns={"si": "si_open", "ic50": "ic50_open", "cc50": "cc50_open"}
    )
    joined = left.merge(right, on=key, how="inner")
    joined = joined.rename(columns={"SI VACV": "si_competition"})
    joined["is_sentinel"] = joined["si_competition"] >= SI_SENTINEL
    joined["log10_error"] = np.log10(joined["si_open"]) - np.log10(joined["si_competition"])
    return joined


def summarise_agreement(joined: pd.DataFrame) -> dict:
    usable = joined[
        ~joined["is_sentinel"]
        & joined["si_competition"].gt(0)
        & joined["si_open"].gt(0)
        & np.isfinite(joined["log10_error"])
    ]
    if usable.empty:
        return {"compared": 0}
    errors = usable["log10_error"]
    return {
        "compared": len(usable),
        "sentinel_rows_excluded": int(joined["is_sentinel"].sum()),
        "median_log10_error": float(errors.median()),
        "median_absolute_log10_error": float(errors.abs().median()),
        "within_2x": float((errors.abs() <= np.log10(2)).mean()),
        "within_5x": float((errors.abs() <= np.log10(5)).mean()),
        "within_10x": float((errors.abs() <= 1.0).mean()),
        "spearman": float(usable["si_open"].corr(usable["si_competition"], method="spearman")),
        "pearson_log": float(
            np.log10(usable["si_open"]).corr(np.log10(usable["si_competition"]))
        ),
    }


def covering_documents(splits: pd.DataFrame, curated_open: pd.DataFrame, key: str = "inchikey14") -> pd.DataFrame:
    """Which publications in the open data contain the competition compounds."""
    keys = set(splits[key].dropna())
    hits = curated_open[curated_open[key].isin(keys) & curated_open["document_id"].notna()]
    grouped = hits.groupby("document_id").agg(
        compounds=(key, "nunique"),
        measurements=(key, "size"),
        doi=("doi", lambda s: next((x for x in s.dropna()), None)),
        year=("year", "max"),
    )
    return grouped.sort_values("compounds", ascending=False).reset_index()


def analyse(curated: pd.DataFrame) -> dict:
    curated_open = open_source_rows(curated)
    tables = build(curated_open)
    paired = tables["relaxed"]

    splits = {name: load_split(name) for name in ("train", "test") if (KAGGLE / f"{name}.csv").exists()}
    overlaps = []
    for split in splits.values():
        for key in ("inchikey", "inchikey14"):
            overlaps.append(overlap(split, curated_open, paired, key))

    joined = agreement(splits["train"], paired) if "train" in splits else pd.DataFrame()
    return {
        "overlaps": overlaps,
        "agreement": joined,
        "agreement_summary": summarise_agreement(joined) if not joined.empty else {"compared": 0},
        "documents": covering_documents(pd.concat(splits.values(), ignore_index=True), curated_open)
        if splits
        else pd.DataFrame(),
        "paired": paired,
        "splits": splits,
    }
