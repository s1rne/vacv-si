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

#: The best single number a model can predict for every row, measured on the
#: training split: mean(`SI VACV`). It is the constant the competition metric
#: rewards in the absence of any information, so it is the thing a
#: substitution has to beat.
BASELINE_SI = 2744.65

#: What the model this dataset feeds already achieves on the competition
#: metric, measured on the full test split. A substitution has to beat *this*,
#: not the constant: a bucket whose recovered values carry a squared error of
#: 16M would make the submission worse even though it beats a constant's 22M.
MODEL_REFERENCE_MSE = 6_000_000.0

#: A bucket needs this many training rows before its measurement is worth
#: acting on, this share of them within a factor of two before its values are
#: worth substituting, and - where its value is the sentinel rather than a
#: ratio - this precision.
MIN_BUCKET_ROWS = 5
MIN_BUCKET_ACCURACY = 0.5
MIN_SENTINEL_PRECISION = 0.8

#: Within a factor of two is the honest threshold for "the same measurement
#: reported differently"; 0.3 log units is a factor of 2.0.
CLOSE_LOG10 = 0.3


def document_prior(records, split: str = "train") -> dict:
    """Which publication each (strain, Culture) group's rows were taken from.

    Fitted on the training split: a group's row votes for a publication when
    that publication contains a record equal to the row's ``IC50_uM_VACV``
    *and* one equal to its ``CC50_uM``. The organisers' groups are
    paper-shaped, so this resolves the compounds that appear in several
    papers. Votes are stored as the set of compound titles that cast them, so
    :func:`vacv.recover._group_votes` can drop a row's own vote and keep the
    measurement honest.
    """
    from .recover import _key

    frame = load_split(split)
    prior: dict[tuple, dict[str, set]] = {}
    for record in frame.to_dict("records"):
        key = _key(record.get("inchikey14"))
        if not key or pd.isna(record.get("IC50_uM_VACV")) or pd.isna(record.get("CC50_uM")):
            continue
        antiviral = records.antiviral.get(key, [])
        cytotox = records.cytotox.get(key, [])
        shared = {a["document_id"] for a in antiviral if a.get("document_id")} & {
            c["document_id"] for c in cytotox if c.get("document_id")
        }
        for document in shared:
            found_ic = any(
                np.isclose(a["organiser_value"], float(record["IC50_uM_VACV"]), rtol=VALUE_RTOL)
                for a in antiviral
                if a["document_id"] == document
            )
            found_cc = any(
                np.isclose(c["organiser_value"], float(record["CC50_uM"]), rtol=VALUE_RTOL)
                for c in cytotox
                if c["document_id"] == document
            )
            if found_ic and found_cc:
                group = prior.setdefault((record.get("strain"), record.get("Culture")), {})
                group.setdefault(document, set()).add(str(record.get("Title")))
    log.info("document prior: %d (strain, Culture) groups pinned to a publication", len(prior))
    return prior


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


def _squared_error(group: pd.DataFrame, prediction: "np.ndarray") -> float:
    truth = pd.to_numeric(group["si_competition"], errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(truth) & np.isfinite(prediction)
    if not usable.any():
        return float("nan")
    return float(np.mean((prediction[usable] - truth[usable]) ** 2))


def _stats(group: pd.DataFrame) -> dict:
    matched = group[group["si"].notna()]
    comparable = matched[~matched["is_sentinel"] & np.isfinite(matched["log10_error"])]
    errors = comparable["log10_error"].abs()
    called_sentinel = matched[matched["si"].eq(SI_SENTINEL)]
    return {
        "matched": len(matched),
        "sentinel": int(matched["is_sentinel"].sum()),
        "sentinel_share": float(matched["is_sentinel"].mean()) if len(matched) else np.nan,
        "called_sentinel": len(called_sentinel),
        "sentinel_precision": float(called_sentinel["is_sentinel"].mean()) if len(called_sentinel) else np.nan,
        "compared": len(comparable),
        "si_exact": float(comparable["si_exact"].mean()) if len(comparable) else np.nan,
        "both_halves_exact": float(comparable["reproduced"].mean()) if len(comparable) else np.nan,
        "median_abs_log10": float(errors.median()) if len(errors) else np.nan,
        "within_0.3_log": float((errors <= CLOSE_LOG10).mean()) if len(errors) else np.nan,
        "within_1_log": float((errors <= 1.0).mean()) if len(errors) else np.nan,
        # The competition metric, restricted to this group: what substituting
        # the recovered value would cost, against what the best constant
        # costs on the same rows. Substitute a tier only when the first number
        # is the smaller one.
        "mse_substituted": _squared_error(
            matched, np.clip(pd.to_numeric(matched["si"], errors="coerce").to_numpy(dtype=float), 0, SI_SENTINEL)
        ),
        "mse_constant": _squared_error(matched, np.full(len(matched), BASELINE_SI)),
        # The metric is squared error, so the optimal prediction is the
        # conditional mean, not the recovered value: a tier that lands on a
        # sentinel row one time in four should predict a quarter of the way
        # towards 10000 on every row of that tier. The weight is this group's
        # own sentinel share, which makes the number in-sample optimistic by
        # about as much as one parameter can be.
        "mse_blended": _squared_error(
            matched,
            (1 - float(matched["is_sentinel"].mean() if len(matched) else 0))
            * np.clip(pd.to_numeric(matched["si"], errors="coerce").to_numpy(dtype=float), 0, SI_SENTINEL)
            + float(matched["is_sentinel"].mean() if len(matched) else 0) * SI_SENTINEL,
        ),
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


def bucket_report(labelled: pd.DataFrame) -> pd.DataFrame:
    """One row per (match tier, pairing rule): the decision table."""
    matched = labelled[labelled["si"].notna()]
    rows = []
    for (tier, rule), group in matched.groupby(["match_tier", "rule"]):
        rows.append({"match_tier": tier, "rule": rule, **_stats(group)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("matched", ascending=False).reset_index(drop=True)


def report(labelled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Accuracy of the recovered values on train, cut three ways."""
    return {
        "match_tier": by(labelled, "match_tier", MATCH_TIERS),
        "rule": by(labelled, "rule", PAIRING_RULES),
        "bucket": bucket_report(labelled),
        "si_quality": by(
            labelled,
            "si_quality",
            ("measured", "lower_bound", "upper_bound", "undetermined", "reported_ratio", "sentinel"),
        ),
    }


def sentinel_report(labelled: pd.DataFrame) -> pd.DataFrame:
    """How well "a cytotoxicity record and no antiviral record" calls the sentinel."""
    rows = []
    for name, mask in (
        ("si recovered", labelled["si"].notna()),
        ("inactive-against-vaccinia record", labelled["inactive_evidence"]),
        ("cytotox record, no antiviral number", labelled["sentinel_evidence"]),
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


#: A publication needs at least this many training rows before its own
#: reproduction rate is worth more than the global one.
MIN_DOCUMENT_ROWS = 2

#: Below this share of training rows within a factor of two, a publication is
#: not a source to substitute from. The failures are not noise: in the
#: Revankar 1984 series the competition's `IC50_uM_VACV` column holds a
#: dimensionless virus rating rather than a concentration, so every ratio
#: recovered from that paper is meaningless however exactly the identifier
#: matched. That is a property of the document, and excluding the document is
#: the only correct response to it.
MIN_DOCUMENT_ACCURACY = 0.5


def document_report(labelled: pd.DataFrame) -> pd.DataFrame:
    """Reproduction accuracy per source publication, on train.

    Aggregate accuracy hides the failure mode that matters: a single paper
    whose numbers mean something other than what the column says, matched
    confidently and wrong for every compound in it.
    """
    matched = labelled[labelled["si"].notna() & labelled["document_id"].notna()]
    rows = []
    for document, group in matched.groupby("document_id"):
        comparable = group[~group["is_sentinel"] & np.isfinite(group["log10_error"])]
        errors = comparable["log10_error"].abs()
        rows.append(
            {
                "document_id": document,
                "doi": next((d for d in group["doi"].dropna()), None),
                "train_rows": len(group),
                "compared": len(comparable),
                "si_exact": float(comparable["si_exact"].mean()) if len(comparable) else np.nan,
                "within_0.3_log": float((errors <= CLOSE_LOG10).mean()) if len(errors) else np.nan,
                "median_abs_log10": float(errors.median()) if len(errors) else np.nan,
                "verdict": _verdict(len(comparable), errors),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["verdict", "train_rows"], ascending=[True, False]).reset_index(drop=True)


def _verdict(compared: int, errors: pd.Series) -> str:
    if compared < MIN_DOCUMENT_ROWS:
        return "thin"
    return "validated" if (errors <= CLOSE_LOG10).mean() >= MIN_DOCUMENT_ACCURACY else "failed"


def document_trust(lookup: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    """Per-row columns saying what train says about the row's own publication."""
    report = document_report(labelled)
    verdict = dict(zip(report["document_id"], report["verdict"]))
    rows = dict(zip(report["document_id"], report["train_rows"]))
    accuracy = dict(zip(report["document_id"], report["within_0.3_log"]))
    documents = lookup["document_id"]
    return pd.DataFrame(
        {
            "document_verdict": documents.map(verdict).fillna("unseen on train"),
            "document_train_rows": documents.map(rows).fillna(0).astype(int),
            "document_train_within_0.3_log": documents.map(accuracy),
        },
        index=lookup.index,
    )


def public_slice(lookup: pd.DataFrame, rows: int = 179) -> pd.DataFrame:
    """Coverage over the rows the public leaderboard is computed on.

    The public score uses the first rows of `test.csv` by index, so a coverage
    number measured on the whole test file does not predict the public score
    and vice versa. Both are reported.
    """
    test = lookup[lookup["split"].eq("test")].copy()
    test["index"] = pd.to_numeric(test["index"], errors="coerce")
    public = test[test["index"] < rows]
    private = test[test["index"] >= rows]
    out = []
    for name, subset in (("public (first %d)" % rows, public), ("private", private), ("full test", test)):
        matched = subset[subset["si"].notna()]
        out.append(
            {
                "slice": name,
                "compounds": len(subset),
                "si_recovered": len(matched),
                "coverage": len(matched) / len(subset) if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(out)


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


def advice(lookup: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    """Per-row: is this value worth substituting, and for what number?

    Decided by the competition's own metric, measured on train. A bucket -
    one match tier crossed with one pairing rule - earns a substitution when
    the squared error of its recovered values beats the squared error of the
    best constant on the same rows. Two further columns carry the arithmetic
    the metric actually wants: because MSE is minimised by the conditional
    mean, a bucket that lands on a sentinel row one time in four should be
    substituted a quarter of the way towards 10000 rather than at its face
    value.

    A publication whose own training rows failed validation is never
    recommended, whatever its bucket scored.
    """
    matched = labelled[labelled["si"].notna()]
    rates: dict[tuple, dict] = {}
    for bucket, group in matched.groupby(["match_tier", "rule"]):
        stats = _stats(group)
        beats_constant = bool(
            np.isfinite(stats["mse_blended"])
            and np.isfinite(stats["mse_constant"])
            and stats["mse_blended"] < stats["mse_constant"]
        )
        if group["si"].eq(SI_SENTINEL).all():  # noqa: SIM108 - the two branches read better apart
            # The bucket's value is the sentinel itself, so the question is
            # precision, not log error.
            worth_it = bool(stats["sentinel_precision"] >= MIN_SENTINEL_PRECISION)
        else:
            # Beating the constant is not enough on its own: a bucket full of
            # sentinel rows beats it by being dragged towards 10000, which is
            # a classifier's job and not a recovered measurement. The value
            # itself has to be right often enough to be worth substituting.
            worth_it = (
                beats_constant
                and bool(stats["within_0.3_log"] >= MIN_BUCKET_ACCURACY)
                and bool(stats["mse_blended"] < MODEL_REFERENCE_MSE)
            )
        rates[bucket] = {
            "sentinel_share": float(group["is_sentinel"].mean()),
            "worth_it": worth_it,
            "rows": len(group),
        }
    overall = {
        "sentinel_share": float(matched["is_sentinel"].mean()) if len(matched) else np.nan,
        "worth_it": False,
        "rows": 0,
    }

    shares, flags = [], []
    for tier, rule in zip(lookup["match_tier"], lookup["rule"]):
        found = rates.get((tier, rule))
        if found is None or found["rows"] < MIN_BUCKET_ROWS:
            # Nothing on train to judge this bucket by. Unjudged is not
            # trusted: the one bucket that would have been substituted on
            # faith here, flexmatch, is also the worst one measured.
            shares.append(overall["sentinel_share"])
            flags.append(False)
            continue
        shares.append(found["sentinel_share"])
        flags.append(found["worth_it"])
    out = pd.DataFrame({"sentinel_rate_for_bucket": shares, "substitute_recommended": flags}, index=lookup.index)
    if "document_verdict" in lookup.columns:
        out["substitute_recommended"] &= lookup["document_verdict"].ne("failed")
    si = np.clip(pd.to_numeric(lookup["si"], errors="coerce").to_numpy(dtype=float), 0, SI_SENTINEL)
    out["si_blended"] = (1 - out["sentinel_rate_for_bucket"]) * si + out["sentinel_rate_for_bucket"] * SI_SENTINEL
    out.loc[lookup["si"].isna(), ["si_blended", "substitute_recommended"]] = [np.nan, False]
    return out


def coverage(lookup: pd.DataFrame) -> pd.DataFrame:
    """Compounds with a recovered SI per split, tier and quality class."""
    rows = []
    for split, group in lookup.groupby("split"):
        matched = group[group["si"].notna()]
        recommended = (
            group[group["substitute_recommended"].fillna(False).astype(bool)]
            if "substitute_recommended" in group.columns
            else matched.iloc[:0]
        )
        row = {
            "split": split,
            "compounds": len(group),
            "si_recovered": len(matched),
            "coverage": len(matched) / len(group) if len(group) else np.nan,
            "recommended": len(recommended),
            "recommended_coverage": len(recommended) / len(group) if len(group) else np.nan,
            "sentinel_evidence_only": int((group["sentinel_evidence"] & group["si"].isna()).sum()),
        }
        for tier in MATCH_TIERS:
            row[tier] = int(matched["match_tier"].eq(tier).sum())
        for rule in PAIRING_RULES:
            row[rule] = int(matched["rule"].eq(rule).sum())
        for quality in ("measured", "lower_bound", "upper_bound", "undetermined", "reported_ratio", "sentinel"):
            row[quality] = int(matched["si_quality"].eq(quality).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write(lookup: pd.DataFrame, directory) -> "object":
    path = directory / "competition_open_data_lookup.csv"
    lookup.to_csv(path, index=False)
    return path
