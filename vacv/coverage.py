"""Measure the recovered rule, then say how far it reaches.

This is the only module that reads the organisers' ``CC50_uM``,
``IC50_uM_VACV`` and ``SI VACV``, and it reads them from the **training**
split only, to answer two questions:

* does the rule in :mod:`vacv.recover` reproduce their numbers, per match tier
  and per pairing rule, bit for bit;
* and what happens on the rows where it does not - a wrong value substituted
  for a model prediction costs the competition metric far more than a missing
  one.

Both numbers are reported per tier because they differ by an order of
magnitude between tiers, and the modelling side has to decide where to
substitute and where to fall back on the model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .competition import load_split
from .recover import MATCH_TIERS, PAIRING_RULES

log = logging.getLogger(__name__)

#: The organisers' "selectivity could not be determined": the IC50 column is
#: empty because the compound was inactive, so the ratio is not a ratio.
SI_SENTINEL = 10000.0

#: ChEMBL stores the value, the competition file stores the same value; a
#: reproduction is a reproduction only if the two agree to the last digit the
#: file carries. `SI VACV` itself is rounded to three decimals, so it gets a
#: looser tolerance than the two halves it was computed from.
VALUE_RTOL = 1e-6
SI_RTOL = 1e-3

#: Within a factor of two is the honest threshold for "the same measurement
#: reported differently"; 0.3 log units is a factor of 2.0.
CLOSE_LOG10 = 0.3


def join_labels(lookup: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """The recovered rows of one split next to the organisers' own columns.

    Joined positionally, not by key: nine training compounds share a skeleton
    with another row, and a key join would hand one of them the other's
    label.
    """
    truth = load_split(split)
    subset = lookup[lookup["split"].eq(split)].reset_index(drop=True)
    if len(subset) != len(truth) or not (subset["Title"].to_numpy() == truth["Title"].to_numpy()).all():
        raise ValueError(f"recovered table does not line up with {split}.csv")
    out = subset.copy()
    out["si_competition"] = truth["SI VACV"].to_numpy()
    out["ic50_competition"] = truth["IC50_uM_VACV"].to_numpy()
    out["cc50_competition"] = truth["CC50_uM"].to_numpy()
    out["is_sentinel"] = (truth["SI VACV"].eq(SI_SENTINEL) & truth["IC50_uM_VACV"].isna()).to_numpy()
    for side in ("ic50", "cc50"):
        ours = pd.to_numeric(out[side], errors="coerce").to_numpy(dtype=float)
        theirs = pd.to_numeric(out[f"{side}_competition"], errors="coerce").to_numpy(dtype=float)
        out[f"{side}_reproduced"] = np.isclose(ours, theirs, rtol=VALUE_RTOL, equal_nan=False)
    out["reproduced"] = out["ic50_reproduced"] & out["cc50_reproduced"]
    ours = pd.to_numeric(out["si"], errors="coerce").to_numpy(dtype=float)
    theirs = pd.to_numeric(out["si_competition"], errors="coerce").to_numpy(dtype=float)
    out["si_exact"] = np.isclose(ours, theirs, rtol=SI_RTOL, equal_nan=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log10_error"] = np.log10(ours) - np.log10(theirs)
    out.loc[out["is_sentinel"], "log10_error"] = np.nan
    return out


def _stats(group: pd.DataFrame) -> dict:
    matched = group[group["si"].notna()]
    comparable = matched[~matched["is_sentinel"] & np.isfinite(matched["log10_error"])]
    errors = comparable["log10_error"].abs()
    return {
        "matched": len(matched),
        "sentinel": int(matched["is_sentinel"].sum()),
        "sentinel_share": float(matched["is_sentinel"].mean()) if len(matched) else np.nan,
        "compared": len(comparable),
        "si_exact": float(comparable["si_exact"].mean()) if len(comparable) else np.nan,
        "both_halves_exact": float(comparable["reproduced"].mean()) if len(comparable) else np.nan,
        "median_abs_log10": float(errors.median()) if len(errors) else np.nan,
        "within_0.3_log": float((errors <= CLOSE_LOG10).mean()) if len(errors) else np.nan,
        "within_1_log": float((errors <= 1.0).mean()) if len(errors) else np.nan,
    }


def by(labelled: pd.DataFrame, column: str, order: tuple[str, ...] | None = None) -> pd.DataFrame:
    rows = []
    values = list(order) if order else sorted(labelled[column].dropna().unique())
    for value in values:
        group = labelled[labelled[column].eq(value)]
        if group.empty:
            continue
        rows.append({column: value, **_stats(group)})
    total = _stats(labelled)
    rows.append({column: "all", **total})
    return pd.DataFrame(rows)


def report(labelled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Accuracy of the recovered values on train, cut three ways."""
    return {
        "match_tier": by(labelled, "match_tier", MATCH_TIERS),
        "rule": by(labelled, "rule", PAIRING_RULES),
        "si_quality": by(labelled, "si_quality", ("measured", "lower_bound", "upper_bound", "undetermined")),
    }


def sentinel_report(labelled: pd.DataFrame) -> pd.DataFrame:
    """How well "a cytotoxicity record and no antiviral record" calls the sentinel."""
    rows = []
    for name, mask in (
        ("si recovered", labelled["si"].notna()),
        ("cytotox record, no antiviral", labelled["sentinel_evidence"]),
        ("no record of either kind", labelled["n_antiviral_records"].eq(0) & labelled["n_cytotox_records"].eq(0)),
        ("antiviral record exists", labelled["n_antiviral_records"].gt(0)),
    ):
        subset = labelled[mask]
        rows.append(
            {
                "group": name,
                "compounds": len(subset),
                "sentinel_share": float(subset["is_sentinel"].mean()) if len(subset) else np.nan,
            }
        )
    rows.append({"group": "all", "compounds": len(labelled), "sentinel_share": float(labelled["is_sentinel"].mean())})
    return pd.DataFrame(rows)


def coverage_by_sentinel(labelled: pd.DataFrame) -> pd.DataFrame:
    """Train coverage split by whether the organisers' row is a sentinel.

    The two halves of the competition set behave differently and the metric
    punishes them differently. A sentinel row's target is 10000, so a
    recovered ratio substituted there is wrong by four orders of magnitude; a
    live row's target is the ratio itself, so a recovered value is worth
    almost exactly its error. Coverage therefore has to be read per class.
    """
    rows = []
    for name, subset in (
        ("sentinel (SI VACV == 10000)", labelled[labelled["is_sentinel"]]),
        ("live measurement", labelled[~labelled["is_sentinel"]]),
    ):
        matched = subset[subset["si"].notna()]
        errors = matched.loc[np.isfinite(matched["log10_error"]), "log10_error"].abs()
        rows.append(
            {
                "class": name,
                "compounds": len(subset),
                "si_recovered": len(matched),
                "coverage": len(matched) / len(subset) if len(subset) else np.nan,
                "si_exact": float(matched["si_exact"].mean()) if len(matched) else np.nan,
                "within_0.3_log": float((errors <= CLOSE_LOG10).mean()) if len(errors) else np.nan,
                "sentinel_evidence_on_the_rest": int(
                    (subset["si"].isna() & subset["sentinel_evidence"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def confidence(lookup: pd.DataFrame, labelled: pd.DataFrame) -> pd.Series:
    """Per-row probability that the recovered SI is the organisers' own number.

    Estimated on the training split, per (match tier, pairing rule), as the
    share of rows whose value is reproduced within a factor of two *and* is
    not one of the organisers' sentinel rows - a tier that often lands on a
    sentinel row is a tier whose values must not be substituted, however
    accurate its arithmetic is.
    """
    usable = labelled[labelled["si"].notna()].copy()
    usable["good"] = (
        ~usable["is_sentinel"]
        & np.isfinite(usable["log10_error"])
        & (usable["log10_error"].abs() <= CLOSE_LOG10)
    )
    rates = usable.groupby(["match_tier", "rule"])["good"].agg(["mean", "size"])
    overall = float(usable["good"].mean()) if len(usable) else np.nan
    keys = list(zip(lookup["match_tier"], lookup["rule"]))
    values = []
    for key in keys:
        if key in rates.index and rates.loc[key, "size"] >= 5:
            values.append(float(rates.loc[key, "mean"]))
        elif pd.notna(key[0]):
            values.append(overall)
        else:
            values.append(np.nan)
    return pd.Series(values, index=lookup.index, name="confidence")


def coverage(lookup: pd.DataFrame) -> pd.DataFrame:
    """Compounds with a recovered SI per split, tier and quality class."""
    rows = []
    for split, group in lookup.groupby("split"):
        matched = group[group["si"].notna()]
        row = {
            "split": split,
            "compounds": len(group),
            "si_recovered": len(matched),
            "coverage": len(matched) / len(group) if len(group) else np.nan,
            "sentinel_evidence_only": int((group["sentinel_evidence"] & group["si"].isna()).sum()),
        }
        for tier in MATCH_TIERS:
            row[tier] = int(matched["match_tier"].eq(tier).sum())
        for rule in PAIRING_RULES:
            row[rule] = int(matched["rule"].eq(rule).sum())
        for quality in ("measured", "lower_bound", "upper_bound", "undetermined"):
            row[quality] = int(matched["si_quality"].eq(quality).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write(lookup: pd.DataFrame, directory) -> "object":
    path = directory / "competition_open_data_lookup.csv"
    lookup.to_csv(path, index=False)
    return path
