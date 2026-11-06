"""Chemistry-aware curation of the harvested measurement table.

Three jobs, in this order:

1. **Structure standardization.** Public records mix salts, tautomers, charge
   states and mixtures of the same compound, so the raw SMILES is useless as a
   join key. RDKit's ``MolStandardize`` pipeline reduces each record to a
   parent structure and an InChIKey.
2. **Late unit conversion.** A measurement in ug/mL cannot be turned into
   micromolar until the molecular weight is known, and the only trustworthy
   molecular weight is the one computed from the standardized parent. So the
   source modules deliberately leave those rows unconverted and this module
   finishes the job.
3. **Aggregation.** Several papers measure the same compound. Duplicates are
   collapsed by the median of ``log10(value)`` - the median because a single
   mis-transcribed order of magnitude should not move the answer, log space
   because concentrations are log-normal and an arithmetic mean of 0.01 uM and
   100 uM is meaningless.

Every rejection is counted and reported. Nothing is dropped silently.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from .units import MAX_UM, MIN_UM, to_micromolar

log = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

#: Elements a small organic molecule may contain. Anything else (metals,
#: boron clusters, As, Hg, Pt complexes) is out of domain for a descriptor or
#: fingerprint model trained on organics, and its measured CC50 usually
#: reflects metal toxicity rather than structure.
ALLOWED_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I",
}

MIN_MW = 90.0
MAX_MW = 1200.0
MIN_HEAVY_ATOMS = 5
MAX_HEAVY_ATOMS = 100

#: log10 spread above which duplicate measurements of one compound disagree
#: enough to be worth flagging (one order of magnitude).
VARIANCE_FLAG_LOG10 = 1.0


@dataclass
class Standardized:
    smiles: str | None
    inchikey: str | None
    mol_weight: float | None
    reject: str | None


_uncharger = rdMolStandardize.Uncharger()
_fragment_chooser = rdMolStandardize.LargestFragmentChooser()
_cache: dict[str, Standardized] = {}
_tautomer_enumerator: rdMolStandardize.TautomerEnumerator | None = None


def _tautomer() -> rdMolStandardize.TautomerEnumerator:
    global _tautomer_enumerator
    if _tautomer_enumerator is None:
        params = rdMolStandardize.CleanupParameters()
        # Without a cap, a polyhydroxy or polycarbonyl natural product can
        # enumerate tens of thousands of tautomers and dominate the runtime.
        params.maxTautomers = 256
        params.maxTransforms = 256
        # RDKit strips sp3 and bond stereochemistry during tautomer
        # canonicalization by default. That would silently turn every
        # enantiomer pair into one structure in the emitted SMILES, throwing
        # away information a model can use. Deduplication already ignores
        # stereochemistry by using the first 14 InChIKey characters, so there
        # is nothing to gain from destroying it here.
        params.tautomerRemoveSp3Stereo = False
        params.tautomerRemoveBondStereo = False
        params.tautomerRemoveIsotopicHs = False
        _tautomer_enumerator = rdMolStandardize.TautomerEnumerator(params)
    return _tautomer_enumerator


def standardize(smiles: str | None, *, canonical_tautomer: bool = True) -> Standardized:
    if smiles is None or not str(smiles).strip():
        return Standardized(None, None, None, "empty_smiles")
    key = f"{smiles}|{canonical_tautomer}"
    hit = _cache.get(key)
    if hit is not None:
        return hit
    result = _standardize_uncached(str(smiles), canonical_tautomer)
    _cache[key] = result
    return result


def _standardize_uncached(smiles: str, canonical_tautomer: bool) -> Standardized:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return Standardized(None, None, None, "unparsable_smiles")
    try:
        mol = rdMolStandardize.Cleanup(mol)
        # Strips salts, solvents and counter-ions and returns the parent.
        mol = rdMolStandardize.FragmentParent(mol)
        mol = _fragment_chooser.choose(mol)
        mol = _uncharger.uncharge(mol)
        if canonical_tautomer:
            mol = _tautomer().Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except Exception as exc:  # RDKit raises a mix of exception types here
        return Standardized(None, None, None, f"standardize_failed:{type(exc).__name__}")

    if mol.GetNumAtoms() == 0:
        return Standardized(None, None, None, "empty_after_stripping")
    if len(Chem.GetMolFrags(mol)) > 1:
        return Standardized(None, None, None, "mixture")

    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if not elements <= ALLOWED_ELEMENTS:
        return Standardized(None, None, None, "disallowed_elements:" + ",".join(sorted(elements - ALLOWED_ELEMENTS)))
    if not any(atom.GetSymbol() == "C" for atom in mol.GetAtoms()):
        return Standardized(None, None, None, "inorganic")

    heavy = mol.GetNumHeavyAtoms()
    if heavy < MIN_HEAVY_ATOMS:
        return Standardized(None, None, None, "too_few_heavy_atoms")
    if heavy > MAX_HEAVY_ATOMS:
        return Standardized(None, None, None, "too_many_heavy_atoms")

    weight = Descriptors.MolWt(mol)
    if not MIN_MW <= weight <= MAX_MW:
        return Standardized(None, None, None, "mw_out_of_range")

    canonical = Chem.MolToSmiles(mol)
    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey:
        return Standardized(None, None, None, "no_inchikey")
    return Standardized(canonical, inchikey, weight, None)


def curate(frame: pd.DataFrame, *, canonical_tautomer: bool = True) -> tuple[pd.DataFrame, Counter]:
    """Standardize structures, finish unit conversion, drop what cannot be used.

    Returns the surviving rows plus a counter of rejection reasons.
    """
    reasons: Counter = Counter()
    out = frame.copy()

    standardized = [standardize(s, canonical_tautomer=canonical_tautomer) for s in out["smiles"]]
    out["smiles_std"] = [s.smiles for s in standardized]
    out["inchikey"] = [s.inchikey for s in standardized]
    out["inchikey14"] = [s.inchikey[:14] if s.inchikey else None for s in standardized]
    out["mol_weight"] = [s.mol_weight for s in standardized]
    out["curation_reject"] = [s.reject for s in standardized]

    # Second attempt at the mass-concentration rows, now that MW is known.
    out["value_um"] = pd.to_numeric(out["value_um"], errors="coerce").astype("float64")
    out["mol_weight"] = pd.to_numeric(out["mol_weight"], errors="coerce").astype("float64")
    missing = out["value_um"].isna() & out["raw_units"].notna() & out["mol_weight"].notna()
    if missing.any():
        recovered = pd.Series(
            [
                to_micromolar(row.raw_value, row.raw_units, mol_weight=row.mol_weight)[0]
                for row in out.loc[missing].itertuples()
            ],
            index=out.index[missing],
            dtype="float64",
        )
        out.loc[missing, "value_um"] = recovered
        log.info(
            "curate: recovered %d of %d mass-concentration rows",
            int(recovered.notna().sum()),
            int(missing.sum()),
        )

    out.loc[out["curation_reject"].isna() & out["value_um"].isna(), "curation_reject"] = "no_micromolar_value"
    implausible = out["curation_reject"].isna() & ~out["value_um"].between(MIN_UM, MAX_UM)
    out.loc[implausible, "curation_reject"] = "value_outside_plausible_range"
    out.loc[out["curation_reject"].isna() & (out["measurement_class"] == "unknown"), "curation_reject"] = (
        "unclassifiable_endpoint"
    )

    out["censored"] = ~out["relation"].isin(["=", "~"])

    reasons.update(out["curation_reject"].dropna().tolist())
    kept = out[out["curation_reject"].isna()].copy()
    kept["log_um"] = np.log10(kept["value_um"].astype(float))
    log.info("curate: %d of %d rows survive; rejections %s", len(kept), len(out), dict(reasons.most_common(12)))
    return kept.reset_index(drop=True), reasons


#: What a source writes instead of a number when the compound did nothing.
INACTIVE_COMMENTS = (
    "not active",
    "inactive",
    "no activity",
    "not effective",
    "no inhibition",
    "not active at",
)


def inactive_evidence(frame: pd.DataFrame, *, canonical_tautomer: bool = True) -> pd.DataFrame:
    """Records that say "tested against vaccinia, nothing happened".

    These carry no number, so :func:`curate` drops them, and for a dataset of
    measurements that is right. For predicting the competition's column it
    throws away the single most valuable row in the harvest: the organisers'
    ``SI VACV == 10000`` is what their pipeline wrote when the IC50 column was
    empty, and the IC50 column was empty because the public record said "Not
    Active" rather than giving a concentration. One such record is worth more
    than any structural feature, because it is the same fact, from the same
    source, that produced their sentinel.

    Returned separately rather than kept in the curated table, so that nothing
    downstream can mistake a missing value for a measured one.
    """
    from . import classify as classify_module

    out = frame.copy()
    numeric = pd.to_numeric(out["value_um"], errors="coerce")
    columns = [c for c in ("text_value", "activity_comment", "data_validity") if c in out.columns]
    said = pd.Series("", index=out.index, dtype="object")
    for column in columns:
        said = said + " " + out[column].astype("string").fillna("").str.lower()
    says_inactive = said.apply(lambda text: any(word in text for word in INACTIVE_COMMENTS))
    candidates = out[numeric.isna() & says_inactive]
    if candidates.empty:
        return pd.DataFrame(columns=["inchikey", "inchikey14", "document_id", "source_id"])
    keep = [
        classify_module.mentions_vaccinia(row.organism, row.assay_description, row.strain)
        and (
            classify_module.mentions_antiviral(row.assay_description)
            or classify_module.classify(
                row.standard_type, description=row.assay_description, organism=row.organism
            )
            in {"antiviral", "selectivity"}
        )
        for row in candidates.itertuples()
    ]
    candidates = candidates[pd.Series(keep, index=candidates.index)]
    standardized = [standardize(s, canonical_tautomer=canonical_tautomer) for s in candidates["smiles"]]
    candidates = candidates.assign(
        inchikey=[s.inchikey for s in standardized],
        inchikey14=[s.inchikey[:14] if s.inchikey else None for s in standardized],
    )
    candidates = candidates[candidates["inchikey14"].notna()]
    log.info(
        "curate: %d records report inactivity against vaccinia without a number (%d compounds)",
        len(candidates),
        candidates["inchikey14"].nunique(),
    )
    columns = [
        "inchikey",
        "inchikey14",
        "standard_type",
        "text_value",
        "activity_comment",
        "data_validity",
        "assay_description",
        "document_id",
        "doi",
        "year",
        "cell_line",
        "source_db",
        "source_id",
    ]
    return candidates[[c for c in columns if c in candidates.columns]].reset_index(drop=True)


def aggregate(
    frame: pd.DataFrame,
    group_columns: list[str],
    *,
    value_column: str = "value_um",
) -> pd.DataFrame:
    """Median-in-log-space collapse of duplicate measurements.

    ``n_measurements`` and ``log10_spread`` travel with every aggregated row;
    ``high_variance`` marks the compounds whose replicate measurements
    disagree by more than an order of magnitude. Those rows are kept, not
    dropped: for most of them the disagreement is a real difference in assay
    conditions, and dropping them would preferentially remove exactly the
    well-studied compounds.
    """
    numeric = frame.copy()
    numeric["_log"] = np.log10(numeric[value_column].astype(float))

    def column(group: pd.DataFrame, name: str) -> pd.Series:
        """A column, or an empty series when it was consumed as a group key."""
        if name in group.columns:
            return group[name]
        return pd.Series(dtype="object")

    def collapse(group: pd.DataFrame) -> pd.Series:
        logs = group["_log"].to_numpy(dtype=float)
        spread = float(logs.max() - logs.min()) if len(logs) > 1 else 0.0
        median = float(np.median(logs))
        uncensored = group[~group["censored"]]
        return pd.Series(
            {
                value_column: 10.0**median,
                "n_measurements": len(group),
                "n_uncensored": int(len(uncensored)),
                "log10_spread": spread,
                "high_variance": spread > VARIANCE_FLAG_LOG10,
                "relation": "=" if len(uncensored) else group["relation"].mode().iat[0],
                "standard_type": "|".join(sorted(set(group["standard_type"].dropna()))),
                "cell_line": "|".join(sorted(set(column(group, "cell_line").dropna()))) or None,
                "strain": "|".join(sorted(set(column(group, "strain").dropna()))) or None,
                "doi": "|".join(sorted(set(column(group, "doi").dropna()))) or None,
                "source_db": "|".join(sorted(set(group["source_db"].dropna()))),
                "source_id": "|".join(sorted(set(group["source_id"].dropna()))[:20]),
                "document_id": "|".join(sorted(set(column(group, "document_id").dropna()))[:20]) or None,
                "year": group["year"].dropna().max() if group["year"].notna().any() else pd.NA,
                "assay_description": group["assay_description"].dropna().iat[0]
                if group["assay_description"].notna().any()
                else None,
                "smiles_std": group["smiles_std"].iat[0],
                "inchikey": group["inchikey"].iat[0],
                "mol_weight": float(group["mol_weight"].iat[0]),
            }
        )

    def collapse_without_keys(group: pd.DataFrame) -> pd.Series:
        # Group keys come back via reset_index, so re-emitting them here would
        # collide on insert.
        return collapse(group).drop(labels=group_columns, errors="ignore")

    collapsed = numeric.groupby(group_columns, dropna=False).apply(collapse_without_keys, include_groups=False)
    return collapsed.reset_index()
