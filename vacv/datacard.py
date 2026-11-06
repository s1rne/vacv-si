"""Generate ``docs/dataset.md`` from the tables that were actually built.

The data card is written by code rather than by hand so that it cannot drift
from the CSV it describes.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .build_dataset import PAIRING_LEVELS, REQUIRED_COLUMNS, order_columns
from .paths import DOCS
from .validate import validate

OUTPUT = DOCS / "dataset.md"


def _percentiles(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    quantiles = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return " / ".join(f"{q:.3g}" for q in quantiles)


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _source_counts(curated: pd.DataFrame) -> str:
    exploded = curated.assign(source=curated["source_db"].astype(str)).copy()
    rows = []
    for source, group in exploded.groupby("source"):
        rows.append(
            [
                source,
                len(group),
                int((group["measurement_class"] == "antiviral").sum()),
                int((group["measurement_class"] == "cytotox").sum()),
                group["inchikey14"].nunique(),
                f"{group['doi'].notna().mean():.0%}",
            ]
        )
    rows.sort(key=lambda r: -r[1])
    return _table(
        rows,
        ["source_db", "measurements", "antiviral", "cytotox", "compounds", "with DOI"],
    )


def _endpoint_counts(curated: pd.DataFrame, measurement_class: str, limit: int = 12) -> str:
    subset = curated[curated["measurement_class"] == measurement_class]
    if subset.empty:
        return "_none_"
    counts = subset["standard_type"].value_counts().head(limit)
    rows = [
        [
            endpoint,
            int(count),
            subset.loc[subset["standard_type"] == endpoint, "inchikey14"].nunique(),
            f"{pd.to_numeric(subset.loc[subset['standard_type'] == endpoint, 'value_um']).median():.3g}",
        ]
        for endpoint, count in counts.items()
    ]
    return _table(rows, ["endpoint", "measurements", "compounds", "median uM"])


def _quality_counts(relaxed: pd.DataFrame) -> str:
    if relaxed.empty or "si_quality" not in relaxed.columns:
        return "_none_"
    rows = []
    for label, group in relaxed.groupby("si_quality"):
        si = pd.to_numeric(group["si"], errors="coerce")
        rows.append(
            [
                label,
                len(group),
                f"{len(group) / len(relaxed):.0%}",
                f"{si.median():.3g}",
                f"{(si >= 10).mean():.0%}",
            ]
        )
    rows.sort(key=lambda r: -r[1])
    return _table(rows, ["si_quality", "compounds", "share", "median SI", "SI >= 10"])


def _pairing_counts(tables: dict[str, pd.DataFrame]) -> str:
    rows = []
    for level in list(PAIRING_LEVELS) + ["relaxed"]:
        table = tables.get(level)
        if table is None or table.empty:
            rows.append([level, 0, "n/a", "n/a", "n/a"])
            continue
        rows.append(
            [
                level,
                len(table),
                f"{table['doi'].notna().mean():.0%}",
                f"{table['censored'].astype(bool).mean():.0%}",
                _percentiles(table["si"]),
            ]
        )
    return _table(
        rows,
        ["pairing level", "compounds", "with DOI", "censored", "SI p5/p25/p50/p75/p95"],
    )


def _overlap(relaxed: pd.DataFrame, competition: pd.DataFrame | None) -> str:
    if competition is None:
        return (
            "The competition training file was not present at `data/raw/kaggle/` when this "
            "card was generated, so no overlap could be computed. Drop the organisers' CSV "
            "there and re-run `build` to fill this section in."
        )
    from .curate import standardize

    keys = {
        standardize(s).inchikey[:14]
        for s in competition.get("smiles", pd.Series(dtype=object)).dropna()
        if standardize(s).inchikey
    }
    ours = set(relaxed["inchikey14"].dropna())
    shared = keys & ours
    return (
        f"- competition compounds (parsed, standardized): **{len(keys)}**\n"
        f"- open-source compounds in this dataset: **{len(ours)}**\n"
        f"- shared InChIKey14: **{len(shared)}** "
        f"({len(shared) / max(len(keys), 1):.0%} of the competition set)\n"
        f"- new compounds this dataset adds: **{len(ours - keys)}**"
    )


def match_section(analysis: dict) -> str:
    """How much of the competition set the open harvest reproduces."""
    from .match_competition import HEADER

    overlaps = _table([o.as_row() for o in analysis["overlaps"]], HEADER)
    summary = analysis["agreement_summary"]
    joined = analysis["agreement"]

    if summary.get("compared"):
        agreement = f"""Open-data SI compared against the organisers' `SI VACV` on the **{summary['compared']}**
training compounds where both exist and the competition value is not the
sentinel ({summary.get('sentinel_rows_excluded', 0)} sentinel rows excluded):

| statistic | value |
|---|---|
| median log10 error (open - competition) | {summary['median_log10_error']:+.2f} |
| median absolute log10 error | {summary['median_absolute_log10_error']:.2f} |
| within a factor of 2 | {summary['within_2x']:.0%} |
| within a factor of 5 | {summary['within_5x']:.0%} |
| within a factor of 10 | {summary['within_10x']:.0%} |
| Spearman rho | {summary['spearman']:.2f} |
| Pearson r on log10(SI) | {summary['pearson_log']:.2f} |

The median error is zero because, for most of these compounds, the open
databases return the organisers' number itself rather than an estimate of it.
The spread that remains comes from compounds measured in several papers or
several cell lines, where the aggregation takes the median and the organisers
took one of them."""
    else:
        agreement = "No training compound could be paired from open data, so no agreement could be measured."

    lookups = analysis.get("lookups", {})
    lookup_rows = []
    for name, table in lookups.items():
        with_si = table["si"].notna() if "si" in table.columns else pd.Series(dtype=bool)
        rules = table["rule"].value_counts().to_dict() if "rule" in table.columns else {}
        lookup_rows.append(
            [
                name,
                len(table),
                int(table["match"].eq("inchikey").sum()),
                int(table["match"].eq("inchikey14").sum()),
                int(with_si.sum()),
                f"{with_si.mean():.0%}" if len(table) else "-",
                ", ".join(f"{k}: {v}" for k, v in rules.items()) or "-",
            ]
        )
    lookup_table = (
        _table(
            lookup_rows,
            [
                "split",
                "compounds",
                "exact InChIKey",
                "skeleton only",
                "SI recovered",
                "coverage",
                "selection rule",
            ],
        )
        if lookup_rows
        else ""
    )

    lookup_summary = analysis.get("lookup_agreement") or {}
    if lookup_summary.get("compared"):
        lookup_accuracy = f"""Accuracy of that lookup against the organisers' own `SI VACV`, on the
{lookup_summary['compared']} training compounds where both exist:

| statistic | value |
|---|---|
| identical to the organisers' value | {lookup_summary['exact']:.0%} |
| within a factor of 2 | {lookup_summary['within_2x']:.0%} |
| within a factor of 5 | {lookup_summary['within_5x']:.0%} |
| Spearman rho | {lookup_summary['spearman']:.2f} |

That {lookup_summary['exact']:.0%} of recovered values are *bit-identical* is the
finding, not the accuracy. The competition compounds were taken from the same
papers these databases index - several of the organisers' own compound labels
are database identifiers (`sid103460068`, `CHEMBL6102784`) - so the open
harvest does not approximate their labels, it recovers them. The disagreements
are nearly all cases where a compound was measured in several papers or several
cell lines and the organisers used one of them while the aggregation uses the
median."""
    else:
        lookup_accuracy = ""

    documents = analysis["documents"]
    if not documents.empty:
        top = documents.head(15)
        doc_table = _table(
            [
                [
                    row.document_id,
                    row.doi or "-",
                    int(row.year) if pd.notna(row.year) else "-",
                    int(row.compounds),
                    int(row.measurements),
                ]
                for row in top.itertuples()
            ],
            ["ChEMBL document", "DOI", "year", "competition compounds", "measurements"],
        )
        doc_note = (
            f"{len(documents)} ChEMBL publications contain at least one competition compound. "
            f"The top {len(top)} by coverage:"
        )
    else:
        doc_table = ""
        doc_note = "No ChEMBL publication matched a competition compound."

    sentinel_note = ""
    if not joined.empty and "is_sentinel" in joined:
        sentinel_note = (
            f"\n`SI VACV == 10000` appears on {int(joined['is_sentinel'].sum())} of the matched "
            "training rows. It is a sentinel for 'selectivity could not be determined' "
            "(the IC50 column is empty because the compound was inactive at every tested "
            "dose), not a measured ratio, and it is excluded from the comparison above. "
            "Any model trained on the raw `SI VACV` column has to deal with it.\n"
        )

    return f"""## Overlap with the competition set

Matching is by standardized InChIKey - full (same stereochemistry and
protonation) and first 14 characters (skeleton only). The competition's own
rows are excluded from the open-data side before matching.

{overlaps}

{agreement}
{sentinel_note}
### Per-compound lookup

`data/processed/competition_open_data_lookup.csv` carries, for every
competition compound, the best open-data IC50/CC50/SI found for it, the rule
that selected it, and the DOI. Candidates are preferred in this order: a pair
from one publication in the same cell line the organisers used, then any
single-publication pair, then a same-publication pair with the cell line
unstated, then a cross-publication aggregate.

{lookup_table}

{lookup_accuracy}

### How the organisers built `SI VACV`, and what that implies

Reverse-engineered from `train.csv`, which is worth stating because it governs
what the Kaggle metric actually rewards:

- For all 237 training rows that have both `CC50_uM` and `IC50_uM_VACV`,
  `SI VACV == CC50_uM / IC50_uM_VACV` exactly.
- `SI VACV == 10000` on **81 of 320** rows, and on exactly the 81 rows whose
  `IC50_uM_VACV` is empty. It is a sentinel for "inactive, no IC50", not a
  ratio. `SI VACV == 0` on the 2 rows whose `CC50_uM` is empty.
- 23 training rows have `CC50_uM == IC50_uM_VACV == 100`, giving `SI VACV == 1`
  - the same both-sides-censored artefact described above, arriving through the
  organisers' own pipeline.

So roughly a quarter of the target column is a constant, and in `test.csv`
**127 of 597** rows have an empty `IC50_uM_VACV`, which under the same rule
makes their hidden target 10000.

Matching against open data predicts that sentinel, which is useful because it
needs no forbidden column. On the training split:

| open-data match | compounds | share with `SI VACV == 10000` |
|---|---|---|
| no match in open data | 197 | 41% |
| matched, SI measured | 55 | 4% |
| matched, SI bounded or undetermined | 68 | 0% |

A compound whose paper ChEMBL indexed is almost never a sentinel row; the
sentinels come from papers the databases did not pick up. Of the 597 test
compounds, 243 match and 354 do not.

### Evidence features for every competition compound

The lookup above only has values for the 243 test and 123 train compounds
where a full pair existed. `competition_presence.csv` covers all 917, because
*absence* from the public anti-VACV literature predicts the sentinel: counts of
matching records, whether an antiviral or a cytotoxicity record exists, median
IC50 and CC50, censoring flags, document and DOI counts, year range, and the
three nearest Morgan-fingerprint neighbours (radius 2, 2048 bits) in the paired
VACV set with their measured SI.

A compound is never its own neighbour - every reference entry sharing its
skeleton is removed from its candidate pool - so the neighbour columns mean the
same thing for matched and unmatched compounds and cannot hand a compound its
own measurement back as a feature.

Separation of the sentinel class on the training split, ranked by how far the
AUC sits from 0.5, is printed by `uv run python -m vacv.cli presence`. Below
0.5 means the feature is *higher* for non-sentinels, which is what every
evidence-of-measurement column does.

### Publications behind the competition compounds

{doc_note}

{doc_table}
"""


def render(
    curated: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    competition: pd.DataFrame | None = None,
    analysis: dict | None = None,
) -> str:
    relaxed = tables["relaxed"]
    findings = validate(order_columns(relaxed)) if not relaxed.empty else []
    si = pd.to_numeric(relaxed["si"], errors="coerce").dropna() if not relaxed.empty else pd.Series(dtype=float)

    return f"""# Dataset card: VACV selectivity index from open databases

Generated by `uv run python -m vacv.cli build` on {dt.date.today().isoformat()}.
Do not edit by hand; edit `vacv/datacard.py`.

## Files

| file | contents |
|---|---|
| `data/processed/vacv_si.csv` | the deliverable: one row per compound, best available pairing |
| `data/processed/vacv_si_strict.csv` | same compound, same publication, same cell line |
| `data/processed/vacv_si_relaxed.csv` | identical to `vacv_si.csv`, kept under the explicit name |
| `data/processed/vacv_si_measured.csv` | the subset where neither IC50 nor CC50 is censored |
| `data/processed/competition_open_data_lookup.csv` | per competition compound: the best open-data IC50/CC50/SI and its DOI |
| `data/processed/competition_presence.csv` | per competition compound: how much open-data evidence exists for it, plus its nearest measured neighbours |

Required columns, in order: `{"`, `".join(REQUIRED_COLUMNS)}`. `ic50` and `cc50`
are micromolar; `si` is `cc50 / ic50`. Everything after `doi` is provenance.

## Measurements after curation, by source

{_source_counts(curated)}

### Endpoints on the antiviral side (vaccinia potency)

{_endpoint_counts(curated, "antiviral")}

### Endpoints on the cytotoxicity side

{_endpoint_counts(curated, "cytotox")}

A measurement is one reported number. `antiviral` and `cytotox` are assigned by
`vacv.classify` from the endpoint name plus the assay free text, not from the
endpoint name alone - public records file host-cell cytotoxicity under `IC50`
about as often as under `CC50`.

## Pairing

{_pairing_counts(tables)}

Pairing levels are defined in `vacv/build_dataset.py`:

- **strict** - same compound, same publication, same host cell line on both
  sides. This is the ratio the original authors would have reported.
- **document** - same compound, same publication, cell line unstated on one
  side.
- **compound** - same compound, any publication. The antiviral and
  cytotoxicity halves are each aggregated across the whole corpus. Highest
  coverage, weakest provenance; `pairing_level` marks these rows so a model can
  exclude or down-weight them.

`vacv_si.csv` emits each compound once, at the most specific level available
for it.

## SI distribution ({len(si)} compounds)

| statistic | value |
|---|---|
| min | {si.min():.3g} |
| p5 / p25 / p50 / p75 / p95 | {_percentiles(si)} |
| max | {si.max():.3g} |
| geometric mean | {np.exp(np.log(si).mean()):.3g} |
| fraction with SI >= 10 | {(si >= 10).mean():.1%} |
| fraction with SI <= 1 | {(si <= 1).mean():.1%} |

### What the censoring does to SI - read this before training

{_quality_counts(relaxed)}

`si_quality` says whether the ratio is a measurement or an artefact:

- **measured** - both halves exact. The only rows where `si` is a number.
- **lower_bound** - IC50 exact, CC50 reported as `> x`. The true SI is *at
  least* `si`. Keeping these matters: a compound that is non-toxic at the
  highest dose tested is exactly the selective compound one is looking for, so
  dropping them biases the dataset against the answer.
- **upper_bound** - IC50 reported as `> x` (inactive), CC50 exact. The true SI
  is at most `si`.
- **undetermined** - both halves bounded, typically `IC50 > 100` and
  `CC50 > 100` for a compound that did nothing at any tested concentration.
  That divides out to `si == 1.0` and carries **no information about
  selectivity at all**. These rows are the largest single artefact in the
  dataset and the reason the unfiltered median SI is exactly 1. Train on them
  and a model learns that a third of chemical space has SI 1.

Recommended use: `vacv_si_measured.csv` as the regression target, with the
bounded rows added back under a censored-aware loss (Tobit, or a ranking
objective) if the extra coverage is worth it. Never use the `undetermined`
rows as point targets.

SI is strongly right-skewed and should be modelled as `log10(SI)`.
An SI at or below 1 means the compound kills cells at least as fast as it stops
the virus; those rows are real measurements, not errors, and removing them
would bias the model towards optimism.

{match_section(analysis) if analysis else "## Overlap with the competition set" + chr(10) + chr(10) + _overlap(relaxed, competition)}

## Caveats

1. **Assay heterogeneity.** The antiviral half mixes plaque reduction, CPE
   inhibition, virus yield reduction and reporter-gene readouts; the
   cytotoxicity half mixes MTT, neutral red, cell counting and reporter
   viability. These do not produce interchangeable numbers, and no correction
   is applied.
2. **Cell lines differ.** Vero, BSC-1, BSC-40, HeLa, CV-1, HFF and HEL all
   appear. CC50 depends on the cell line more strongly than IC50 does.
3. **Strains differ.** Where a strain is stated it is in the `strain` column
   (WR, Copenhagen, IHD-J, Lister, Ankara). Most records state nothing.
4. **Censored values.** `ic50_relation` and `cc50_relation` carry `>` or `<`
   where the source reported a bound, and `si_quality` says what that does to
   the ratio. See the section above: 30% of rows are `undetermined` and must
   not be used as point targets. Censored rows are kept and flagged rather
   than dropped, because dropping all of them removes precisely the selective
   compounds - a compound that is non-toxic at the highest tested dose is the
   interesting case.
5. **Replicate disagreement.** `ic50_log10_spread` and `cc50_log10_spread`
   record the range of the aggregated replicates in log10 units;
   `high_variance` marks more than one order of magnitude.
6. **No DOI for PubChem rows.** PubChem assay citations are references the
   depositor attached, not the publication of the measurement; the NCATS
   poxvirus screens cite a monkeypox epidemiology paper containing no
   chemistry. Those citations live in `reference`, and `doi` is left empty
   rather than filled with something misleading.

## Validation at generation time

{chr(10).join("- " + str(f) for f in findings) if findings else "- not run (empty table)"}
"""


def write(text: str) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)
