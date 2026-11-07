# VACV selectivity index: the data layer

Builds a training dataset for predicting the selectivity index
(SI = CC50 / IC50) of small molecules against vaccinia virus, assembled from
open databases, with a DOI on every record that has one.

## Current Kaggle target: train_v2

**Use the current Kaggle ZIP.** Its `train_v2.csv` changes 81 old targets from
10000 to zero while preserving the allowed features. Historical sentinel
classifiers optimize the wrong target; previous CV scores are not comparable.
The old submission/CV commands now require `--allow-legacy-target` for historical
replay. They must not be used for current competition submissions.

```sh
python scripts/import_competition_v2.py '/path/to/current-kaggle.zip' --out data/raw_v2
PYTHONPATH=. python scripts/submit_v2.py --data-dir data/raw_v2 --out outputs/v2/submission.csv
python -m pytest -q
```

The importer requires the officially named `train_v2.csv`, reads only permitted
fields, records source hashes, and preserves existing datasets. The new model
regresses SI directly (zero included); it does not predict toward a 10000
sentinel. `experiments/restore_baseline.py`, `experiments/v2_embeddings.py`, and
`experiments/v2_external.py` compare raw/log regression, frozen chemistry,
external record features, and disjoint measured external SI training.

See `experiments/results/target-v2/` for the target audit and current comparison.

### Submission budget and next-model gate

Keep the accepted public best (2,038,068.29444). Further Kaggle submissions are
paused until a frozen candidate achieves full raw-MSE at most 3,000,000 on
each of two confirmation splits, using all 320 official v2 labels. Scaffold
results must also be reported. Do not remove, cap or replace the maximum
training label to pass this gate. Small CV changes do not justify uploads.

The context-matched analog experiment improved paired random CV by 5.43%
(4.258M to 4.027M), but failed the gate and worsened scaffold CV. It has not
been submitted. See `experiments/results/v2-context-analog/` for OOF results,
the exact production baseline, activity-cliff diagnostics, and the 3.016M
mathematical lower bound for train-only predictors bounded by training labels.
Other completed alternatives are recorded under `v2-analog`, `v2-tabicl`,
`v2-hurdle`, and `v2-tweedie`. Tweedie scored 2,062,945.32569 publicly and did
not improve the best.

The subsequent source-record audit found that the external data producer had
pooled MCC with CC50 and censored limits with exact measurements. Do not treat
its aggregate-derived corrections as verified unit fixes. The separate offline
builder `experiments/v3_source_pairs.py` preserves individual activity/assay
records and produces 47 exact SI pairs plus 123 interval observations, with
competition molecules excluded. Missing exposure metadata remains a stated
limitation. See `docs/external-v3-audit.md` and
`experiments/results/v3-source-pairs/` for evidence and provenance.

`experiments/v3_intervals.py` compares exact-only and interval-aware training;
`experiments/v3_distributional.py` tests a lognormal mean correction using
inner-OOF residuals. Both failed the local submission gate; their saved reports
include negative results. Neither experiment submitted predictions to Kaggle.
The final raw/sqrt interval ablation (`experiments/v3_interval_scales.py`)
improved its paired sqrt baseline by only 0.196% on confirmation seeds 28/29:
4.230480M and 4.236306M. It also failed the gate and produced no candidate CSV.

## Recovery checkpoint (2026-11-07)

Model CV and submission commands now default to `--lookup-policy clean`: allowed
structure/context plus external presence counts and similarities, with **no SI
lookup substitution**. This avoids the saved lookup's full-training-label source
selection. Competition concentration columns are excluded at CSV read time.
`--lookup-policy recovered` is an explicit legacy inference option; for CV it
requires the original external corpus and reconstructs matches within each fold.
Missing corpus files raise an error, rather than returning a contaminated score.

```sh
PYTHONPATH=. uv run python scripts/cv_checkpoint.py --splits 6 7 8 --out outputs/clean-cv
PYTHONPATH=. uv run python scripts/submit_checkpoint.py --out outputs/clean/submission.csv
# Lightweight calibrated hybrid, no pretrained weights needed:
PYTHONPATH=. uv run python scripts/submit_recovery.py --out outputs/fallback/submission.csv
```

Checkpoint dependencies/cache setup remains in
`experiments/results/checkpoint-2026-11-07/report.md`. New measured comparisons
and submission hashes are in `experiments/results/recovery-clean/`. Earlier
lookup-based CV and reconstructed test-label scores must not be treated as
independent leaderboard evidence. Public score changes for already uploaded
files require checking the competition data/evaluation history.

```sh
uv sync
uv run python -m vacv.cli all                   # fetch -> curate -> build -> validate
uv run pytest                                   # 94 tests
PYTHONPATH=. uv run python scripts/cv.py        # cross-validation, competition metric
PYTHONPATH=. uv run python scripts/submit.py    # outputs/submission.csv, byte-reproducible
```

`data/processed/` is tracked, so the model reproduces without re-running the
harvest; `vacv.cli all` needs network access and a few hours of API calls.
`docs/validation.md` holds the validation record: adversarial validation, the
leakage audit, the applicability domain, uncertainty and the limitations list.

Outputs:

| path | contents |
|---|---|
| `data/raw/<source>/` | untouched API responses, cached by request URL (gitignored) |
| `data/interim/measurements.csv.gz` | one row per reported measurement, all sources |
| `data/interim/curated.csv.gz` | the same after RDKit standardization and unit conversion |
| `data/processed/vacv_si.csv` | the deliverable: `smiles, ic50, cc50, si, doi` plus provenance |
| `data/processed/vacv_si_strict.csv` | pairs from one publication and one cell line only |
| `data/processed/vacv_si_measured.csv` | the subset with neither side censored - the one to train on |
| `data/interim/competition_molecules.csv` | competition compound -> ChEMBL molecule, by identifier, InChIKey or flexmatch |
| `data/interim/inactive.csv.gz` | records that name vaccinia and report "Not active" instead of a number |
| `data/processed/competition_open_data_lookup.csv` | per competition compound: the organisers' own IC50/CC50/SI, recovered |
| `data/processed/competition_aggregate_lookup.csv` | the same compounds, aggregated instead of reconstructed |
| `data/processed/competition_presence.csv` | per competition compound: open-data evidence counts and nearest measured neighbours |
| `docs/dataset.md` | data card, generated by the build |
| `docs/data_sources.md` | per-source endpoint, query, licence, counts, caveats |
| `docs/validation.md` | validation record and the honest limitations list |
| `docs/figures/` | figures for the defence |
| `experiments/` | model selection, one script per question |
| `analysis/` | checks behind `docs/validation.md` |

Stages, each reading the previous one's file from disk:

| command | what it does |
|---|---|
| `vacv.cli fetch` | queries ChEMBL and PubChem BioAssay, including the harvest keyed on the competition compounds; `--source X`, `--refresh` |
| `vacv.cli curate` | RDKit standardization, unit conversion to uM, rejection accounting |
| `vacv.cli build` | pairs IC50 with CC50, computes SI, writes the CSVs and the data card |
| `vacv.cli match` | overlap with the organisers' files in `data/raw/kaggle/` |
| `vacv.cli presence` | evidence features for every competition compound, and how each separates the sentinel class |
| `vacv.cli coverage` | recovers the organisers' IC50/CC50 per compound and scores the rule on train |
| `vacv.cli validate` | schema, SMILES, units, duplicates, SI consistency, DOI coverage |

## Recovering the organisers' own numbers

`SI VACV` is not an independent measurement: on the training split it equals
`CC50_uM / IC50_uM_VACV` exactly for every complete row, and 382 of the 917
compounds are titled with a database identifier. `vacv/recover.py` therefore
reconstructs the record the organisers transcribed rather than estimating the
compound's selectivity: one antiviral record and one cytotoxicity record from
the same publication, preferring the cell line the row's `Culture` names, then
the endpoint they copied (`CC50` before `MCC`), then the censored record, with
the value taken as the source printed it - they did not convert `ug.mL-1` to
micromolar. Where a record says "Not active" against vaccinia instead of
giving a number, that is what left their IC50 column empty, and the row's
answer is the 10000 sentinel.

`vacv/coverage.py` is the only module that reads their columns, and only from
`train.csv`. It scores the rule per match tier and pairing rule, marks the
publications whose own training rows fail validation, and writes
`substitute_recommended` and `si_blended` into the lookup so the modelling
side substitutes where the measurement says it pays and falls back on the
model where it does not. `uv run python -m vacv.cli coverage` prints the
tables.

Curation canonicalizes tautomers by default. On the 48 077 unique SMILES of the
current harvest that costs 8.4 ms per structure (about 400 s) against 1.9 ms
(about 92 s) without, so it stays on; `curate --no-tautomer` turns it off.
RDKit's tautomer canonicalizer strips stereochemistry unless told not to, and
`vacv/curate.py` tells it not to.

The competition's own CSVs are never required. Drop `train.csv` into
`data/raw/kaggle/` to enable the overlap report; `--include-competition` adds
its measurements to the interim table, and they are still excluded from
`data/processed/`.

See `docs/data_sources.md` for what each source contributes, which sources were
evaluated and rejected, and the licence of the assembled dataset (CC BY-SA,
inherited from ChEMBL).
