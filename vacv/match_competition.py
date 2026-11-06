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

from . import coverage, recover
from .build_dataset import all_pairs, antiviral_side, build, cytotox_side
from .competition import load_split
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
    # Exact equality: one training row has a genuine SI of 36066.667.
    joined["is_sentinel"] = joined["si_competition"].eq(SI_SENTINEL)
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


#: Competition `Culture` strings mapped onto the cell-line vocabulary of
#: vacv.classify. The organisers write them freely ("cells HEL", "lung
#: fibroblast cells", "HELA cells"), so matching needs the same normaliser the
#: harvest uses.
def culture_cell_line(culture: object) -> str | None:
    from .classify import extract_cell_line

    return extract_cell_line(culture)


#: Tie-break inside one provenance rule: an exact ratio beats a bounded one.
QUALITY_RANK = {"measured": 0, "lower_bound": 1, "upper_bound": 2, "undetermined": 3}

#: Preference order when several open-data pairs exist for one compound. The
#: first rule reproduces the authors' own ratio in the authors' own cells.
LOOKUP_PRIORITY = [
    ("strict, same cell line", lambda pairs, cell: pairs[pairs["pairing_level"].eq("strict") & pairs["cell_line"].eq(cell)]),
    ("strict", lambda pairs, cell: pairs[pairs["pairing_level"].eq("strict")]),
    ("document", lambda pairs, cell: pairs[pairs["pairing_level"].eq("document")]),
    ("compound", lambda pairs, cell: pairs[pairs["pairing_level"].eq("compound")]),
]


def lookup(split: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Best open-data SI for every compound of a competition split.

    Matching is by exact InChIKey first and by skeleton (first 14 characters)
    only if that fails, because a skeleton match can be a different
    stereoisomer, and stereochemistry changes antiviral potency. The column
    ``match`` records which it was, so a model can weight them differently.
    """
    by_full = {k: g for k, g in pairs.groupby("inchikey")} if "inchikey" in pairs.columns else {}
    by_skeleton = {k: g for k, g in pairs.groupby("inchikey14")}

    rows = []
    # itertuples is not usable here: the competition test file has a column
    # literally named "index", which collides with the namedtuple method.
    for record in split.to_dict("records"):
        cell = culture_cell_line(record.get("Culture"))
        candidates, match = None, None
        if record["inchikey"] in by_full:
            candidates, match = by_full[record["inchikey"]], "inchikey"
        elif record["inchikey14"] in by_skeleton:
            candidates, match = by_skeleton[record["inchikey14"]], "inchikey14"
        row = {
            "index": record.get("index"),
            "Title": record.get("Title"),
            "smiles": record.get("SMILES"),
            "inchikey": record["inchikey"],
            "inchikey14": record["inchikey14"],
            "culture": record.get("Culture"),
            "cell_line": cell,
            "match": match,
        }
        if candidates is not None:
            for rule, selector in LOOKUP_PRIORITY:
                chosen = selector(candidates, cell)
                if not chosen.empty:
                    # Among candidates of equal provenance, an uncensored pair
                    # beats a bounded one. Provenance still comes first: a
                    # same-paper same-cell-line bounded pair is closer to what
                    # the authors reported than an uncensored pair from
                    # somewhere else.
                    if "si_quality" in chosen.columns:
                        chosen = chosen.sort_values(
                            "si_quality",
                            key=lambda s: s.map(QUALITY_RANK).fillna(len(QUALITY_RANK)),
                            kind="stable",
                        )
                    best = chosen.iloc[0]
                    row.update(
                        {
                            "rule": rule,
                            "ic50": float(best["ic50"]),
                            "cc50": float(best["cc50"]),
                            "si": float(best["si"]),
                            "pairing_level": best["pairing_level"],
                            "matched_cell_line": best.get("cell_line"),
                            "censored": bool(best["censored"]),
                            "si_quality": best.get("si_quality"),
                            "doi": best.get("doi"),
                            "document_id": best.get("document_id"),
                        }
                    )
                    break
        rows.append(row)
    return pd.DataFrame(rows)


def lookup_agreement(train_lookup: pd.DataFrame, train: pd.DataFrame) -> dict:
    joined = train_lookup.merge(
        train[["inchikey", "inchikey14", "SI VACV", "Title"]].rename(columns={"SI VACV": "si_competition"}),
        on=["inchikey", "inchikey14", "Title"],
        how="left",
    )
    usable = joined[
        joined["si"].notna()
        & joined["si_competition"].notna()
        & joined["si_competition"].gt(0)
        & joined["si_competition"].ne(SI_SENTINEL)
    ].copy()
    if usable.empty:
        return {"compared": 0}
    usable["log10_error"] = np.log10(usable["si"]) - np.log10(usable["si_competition"])
    errors = usable["log10_error"]
    return {
        "compared": len(usable),
        "exact": float((errors.abs() < 1e-6).mean()),
        "within_2x": float((errors.abs() <= np.log10(2)).mean()),
        "within_5x": float((errors.abs() <= np.log10(5)).mean()),
        "median_absolute_log10_error": float(errors.abs().median()),
        "spearman": float(usable["si"].corr(usable["si_competition"], method="spearman")),
        "by_rule": usable.groupby("rule")["log10_error"].agg(["size", lambda e: float(e.abs().median())]).rename(
            columns={"<lambda_0>": "median_abs_log10_error"}
        ).to_dict("index"),
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
    pairs = all_pairs(curated_open)

    splits = {name: load_split(name) for name in ("train", "test") if (KAGGLE / f"{name}.csv").exists()}
    overlaps = []
    for split in splits.values():
        for key in ("inchikey", "inchikey14"):
            overlaps.append(overlap(split, curated_open, paired, key))

    lookups = {}
    for name, split in splits.items():
        table = lookup(split, pairs)
        table.insert(0, "split", name)
        lookups[name] = table

    joined = agreement(splits["train"], paired) if "train" in splits else pd.DataFrame()
    return {
        **recovery(curated_open, splits),
        "overlaps": overlaps,
        "agreement": joined,
        "agreement_summary": summarise_agreement(joined) if not joined.empty else {"compared": 0},
        "lookups": lookups,
        "lookup_agreement": lookup_agreement(lookups["train"], splits["train"])
        if "train" in lookups
        else {"compared": 0},
        "documents": covering_documents(pd.concat(splits.values(), ignore_index=True), curated_open)
        if splits
        else pd.DataFrame(),
        "paired": paired,
        "splits": splits,
    }


def recovery(curated_open: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> dict:
    """The organisers' own numbers, recovered by the rule of :mod:`vacv.recover`.

    Kept next to the aggregate lookup above rather than replacing it in code,
    because the two answer different questions: the aggregate is the best
    estimate of a compound's selectivity index from everything published about
    it, and this is a reconstruction of the single record the organisers
    transcribed. For predicting their column, the reconstruction wins - see
    the accuracy tables in :mod:`vacv.coverage`.
    """
    if not splits:
        return {"recovered": pd.DataFrame(), "recovery_report": {}, "recovery_coverage": pd.DataFrame()}
    records = recover.Records(curated_open)
    links = recover.molecule_links()
    prior = coverage.document_prior(records) if "train" in splits else {}
    recovered = pd.concat(
        [recover.recover(split, records, links, prior) for split in splits.values()], ignore_index=True
    )
    labelled = coverage.join_labels(recovered, "train") if "train" in splits else pd.DataFrame()
    if not labelled.empty:
        recovered["confidence"] = coverage.confidence(recovered, labelled)
        for column, values in coverage.document_trust(recovered, labelled).items():
            recovered[column] = values
        for column, values in coverage.advice(recovered, labelled).items():
            recovered[column] = values
    return {
        "recovered": recovered,
        "recovery_labelled": labelled,
        "recovery_report": coverage.report(labelled) if not labelled.empty else {},
        "recovery_sentinel": coverage.sentinel_report(labelled) if not labelled.empty else pd.DataFrame(),
        "recovery_by_sentinel": coverage.coverage_by_sentinel(labelled) if not labelled.empty else pd.DataFrame(),
        "recovery_coverage": coverage.coverage(recovered),
        "recovery_documents": coverage.document_report(labelled) if not labelled.empty else pd.DataFrame(),
        "recovery_slices": coverage.public_slice(recovered),
    }


def write_lookups(analysis: dict, directory) -> list:
    """Persist the per-compound lookup so the modelling side can use it."""
    written = []
    recovered = analysis.get("recovered")
    if recovered is not None and not recovered.empty:
        written.append(coverage.write(recovered, directory))
    tables = [t for t in analysis.get("lookups", {}).values()]
    if tables:
        combined = pd.concat(tables, ignore_index=True)
        path = directory / "competition_aggregate_lookup.csv"
        combined.to_csv(path, index=False)
        written.append(path)
    return written
