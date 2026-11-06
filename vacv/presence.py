"""Per-compound evidence features for every competition compound.

The reconstructable-SI lookup covers 243 of 597 test compounds. This table
covers all of them, because *absence* from the public anti-VACV literature is
itself informative: on the training split, compounds with a reconstructable SI
carry the `SI VACV == 10000` sentinel 1.6% of the time and compounds without
one carry it 40.6% of the time. A classifier wants that signal for every row,
not only for the rows where a full pair happened to exist.

Nearest-neighbour columns extend the idea past exact matching: a compound with
no record of its own may still sit next to a well-measured analogue. Neighbours
are taken from the paired VACV set, and a compound is never its own neighbour -
matching skeletons are excluded from its candidate pool, so the column means
the same thing for matched and unmatched compounds and cannot leak a compound's
own measurement back to it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from .build_dataset import antiviral_side, build, cytotox_side
from .match_competition import SI_SENTINEL, load_split, open_source_rows
from .paths import KAGGLE

log = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
NEIGHBOURS = 3

#: Features the sentinel report scores. Booleans and counts alike are scored
#: by rank AUC, which needs no threshold and is comparable across columns.
FEATURE_COLUMNS = [
    "n_records_exact",
    "n_records_skeleton",
    "has_antiviral",
    "has_cytotox",
    "n_antiviral",
    "n_cytotox",
    "best_ic50_um",
    "best_cc50_um",
    "ic50_censored_any",
    "ic50_uncensored_any",
    "cc50_censored_any",
    "cc50_uncensored_any",
    "n_documents",
    "n_dois",
    "earliest_year",
    "latest_year",
    "nn1_tanimoto",
    "nn1_si",
    "nn2_tanimoto",
    "nn2_si",
    "nn3_tanimoto",
    "nn3_si",
]


def _generator() -> rdFingerprintGenerator.FingerprintGenerator64:
    return rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)


def _fingerprint(generator, smiles: object):
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return generator.GetFingerprint(mol)


def _neighbour_columns(
    split: pd.DataFrame, reference: pd.DataFrame, count: int = NEIGHBOURS
) -> pd.DataFrame:
    """Top-``count`` Tanimoto neighbours in the paired VACV set."""
    generator = _generator()
    reference = reference.dropna(subset=["smiles", "si"]).reset_index(drop=True)
    fingerprints, keys, sis, qualities = [], [], [], []
    for row in reference.itertuples():
        fingerprint = _fingerprint(generator, row.smiles)
        if fingerprint is None:
            continue
        fingerprints.append(fingerprint)
        keys.append(row.inchikey14)
        sis.append(float(row.si))
        qualities.append(getattr(row, "si_quality", None))
    keys = np.array(keys, dtype=object)
    sis = np.array(sis, dtype=float)
    qualities = np.array(qualities, dtype=object)

    rows = []
    for record in split.to_dict("records"):
        out: dict = {}
        query = _fingerprint(generator, record.get("SMILES"))
        if query is not None and fingerprints:
            similarity = np.array(DataStructs.BulkTanimotoSimilarity(query, fingerprints))
            # A compound is never its own neighbour: drop every reference entry
            # with the same skeleton, matched or not.
            similarity[keys == record.get("inchikey14")] = -1.0
            order = np.argsort(-similarity)[:count]
            for rank, position in enumerate(order, start=1):
                if similarity[position] < 0:
                    continue
                out[f"nn{rank}_tanimoto"] = float(similarity[position])
                out[f"nn{rank}_si"] = float(sis[position])
                out[f"nn{rank}_inchikey14"] = keys[position]
                out[f"nn{rank}_si_quality"] = qualities[position]
        rows.append(out)
    return pd.DataFrame(rows, index=split.index)


def _record_features(split: pd.DataFrame, curated_open: pd.DataFrame) -> pd.DataFrame:
    antiviral = antiviral_side(curated_open)
    cytotox = cytotox_side(curated_open)

    by_exact = curated_open.groupby("inchikey").size()
    by_skeleton = curated_open.groupby("inchikey14").size()

    def summarise(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        grouped = frame.groupby("inchikey14")
        return pd.DataFrame(
            {
                f"n_{prefix}": grouped.size(),
                f"best_{prefix}_um": grouped["value_um"].median(),
                f"{prefix}_censored_any": grouped["censored"].any(),
                f"{prefix}_uncensored_any": grouped["censored"].apply(lambda s: bool((~s).any())),
            }
        )

    antiviral_summary = summarise(antiviral, "ic50").rename(
        columns={"n_ic50": "n_antiviral"}
    )
    cytotox_summary = summarise(cytotox, "cc50").rename(columns={"n_cc50": "n_cytotox"})

    provenance = curated_open.groupby("inchikey14").agg(
        n_documents=("document_id", "nunique"),
        n_dois=("doi", "nunique"),
        earliest_year=("year", "min"),
        latest_year=("year", "max"),
    )

    out = pd.DataFrame(index=split.index)
    out["n_records_exact"] = split["inchikey"].map(by_exact).fillna(0).astype(int)
    out["n_records_skeleton"] = split["inchikey14"].map(by_skeleton).fillna(0).astype(int)
    for table in (antiviral_summary, cytotox_summary, provenance):
        for column in table.columns:
            out[column] = split["inchikey14"].map(table[column])
    out["n_antiviral"] = out["n_antiviral"].fillna(0).astype(int)
    out["n_cytotox"] = out["n_cytotox"].fillna(0).astype(int)
    for column in (
        "ic50_censored_any",
        "ic50_uncensored_any",
        "cc50_censored_any",
        "cc50_uncensored_any",
    ):
        out[column] = out[column].fillna(False).astype(bool)
    out["has_antiviral"] = out["n_antiviral"].gt(0)
    out["has_cytotox"] = out["n_cytotox"].gt(0)
    out["n_documents"] = out["n_documents"].fillna(0).astype(int)
    out["n_dois"] = out["n_dois"].fillna(0).astype(int)
    return out


def presence_table(curated: pd.DataFrame) -> pd.DataFrame:
    """One row per competition compound, with its open-data evidence."""
    curated_open = open_source_rows(curated)
    reference = build(curated_open)["relaxed"]

    chunks = []
    for name in ("train", "test"):
        if not (KAGGLE / f"{name}.csv").exists():
            continue
        split = load_split(name)
        identity = pd.DataFrame(
            {
                "split": name,
                "index": split["index"] if "index" in split.columns else pd.NA,
                "Title": split["Title"],
                "smiles": split["SMILES"],
                "inchikey": split["inchikey"],
                "inchikey14": split["inchikey14"],
                "culture": split.get("Culture"),
            },
            index=split.index,
        )
        chunks.append(
            pd.concat(
                [
                    identity,
                    _record_features(split, curated_open),
                    _neighbour_columns(split, reference),
                ],
                axis=1,
            )
        )
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    log.info("presence: %d competition compounds, %d with any open record", len(out), int(out["n_records_skeleton"].gt(0).sum()))
    return out


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank AUC, ties averaged. NaN scores are ranked lowest."""
    mask = ~np.isnan(scores)
    if mask.sum() == 0:
        return float("nan")
    filled = np.where(mask, scores, -np.inf)
    positive = labels.astype(bool)
    if positive.sum() == 0 or (~positive).sum() == 0:
        return float("nan")
    order = np.argsort(filled, kind="mergesort")
    ranks = np.empty(len(filled), dtype=float)
    ranks[order] = np.arange(1, len(filled) + 1)
    # average ranks within ties so a constant column scores 0.5
    frame = pd.DataFrame({"value": filled, "rank": ranks})
    ranks = frame.groupby("value")["rank"].transform("mean").to_numpy()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sentinel_report(presence: pd.DataFrame) -> pd.DataFrame:
    """How well each feature separates the `SI VACV == 10000` sentinel class.

    AUC is reported for the sentinel being the positive class, so 0.5 is
    useless and a value far from 0.5 in either direction is informative -
    below 0.5 means the feature is high for non-sentinels, which is what
    evidence-of-measurement columns do.
    """
    train = load_split("train")
    # Exact equality, not a threshold: one training row has a genuine
    # SI VACV of 36066.667 (CC50 1082 / IC50 0.03), so ">= 10000" would
    # mislabel a real measurement as the sentinel.
    sentinel = (train["SI VACV"].eq(SI_SENTINEL) & train["IC50_uM_VACV"].isna()).to_numpy()
    subset = presence[presence["split"].eq("train")].reset_index(drop=True)
    # Positional alignment, not a key join: nine training compounds share a
    # skeleton with another row, and a dict keyed on inchikey14 would hand one
    # of them the other's label.
    if len(subset) != len(train) or not (subset["Title"].to_numpy() == train["Title"].to_numpy()).all():
        raise ValueError("presence table rows do not line up with train.csv")
    labels = sentinel

    rows = []
    for column in FEATURE_COLUMNS:
        if column not in subset.columns:
            continue
        values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=float)
        present = ~np.isnan(values)
        rows.append(
            {
                "feature": column,
                "defined": int(present.sum()),
                "sentinel_median": float(np.nanmedian(values[labels])) if labels.any() and present[labels].any() else np.nan,
                "other_median": float(np.nanmedian(values[~labels])) if (~labels).any() and present[~labels].any() else np.nan,
                "sentinel_rate_if_defined": float(labels[present].mean()) if present.any() else np.nan,
                "sentinel_rate_if_missing": float(labels[~present].mean()) if (~present).any() else np.nan,
                "auc": _auc(labels, values),
            }
        )
    report = pd.DataFrame(rows)
    report["separation"] = (report["auc"] - 0.5).abs()
    return report.sort_values("separation", ascending=False).reset_index(drop=True)


def write(presence: pd.DataFrame, directory):
    path = directory / "competition_presence.csv"
    presence.to_csv(path, index=False)
    return path
