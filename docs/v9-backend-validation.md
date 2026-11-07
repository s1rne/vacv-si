# Model-family comparison and provisional similar-data ablation

User requested testing several model families on the existing data, without
Kaggle submission. This experiment changes no source measurements and predicts
only heldout rows of the official training set.

## Predeclared protocol

- Reuse the exact materialized V8 outer/inner grouped folds. Primary: identity;
  secondary: identity71, identity72, and identity/scaffold connected components.
  These historically reused folds are a benchmark, not a fresh independent holdout.
- Keep all 320 official targets and score pooled raw MSE, including the maximum.
- Baseline: existing submitted v4, unchanged.
- ExtraTrees and CatBoost, each raw and log1p; squared-similarity kernel ridge,
  each raw and log1p. No searches within individual model hyperparameters.
  Exact parameters and library versions are frozen in the protocol JSON.
- Tree/boosting inputs use existing structural features only. Kernel inputs
  use the fixed squared binary similarity. No new experimental-data features,
  forbidden competition columns, label replacement or external query lookup.
- Model selection uses inner OOF only. Core ensemble selects among 28 convex
  weight vectors in halves for the 7 core models. External exploratory ensemble
  selects among 36 vectors for 8 models. Exact ties use baseline-first descending
  lexicographic grid order. Outer labels never enter these choices.
- Primary endpoint: core-blend vs v4 on primary grouped raw MSE. Confirmation
  requires improvement on primary and both repeats, scaffold not worse.
- Data-only comparison: et-log-external vs et-log with identical estimator
  settings. This result and the ensemble including it remain exploratory.
- No test inference, submission generation, upload, or deployment.

## Review of exp/similar

Immutable source commit: `066ad8bf2f48afc9d469a5925cde2ed9676541d0`.
The branch adds data and notes, not evaluated training code. The combined table
contains 588 rows, 250 stored full identity keys and 54 DOI strings. After the
same standardization used by our validation it represents 236 identity groups.

The competition-overlap flag misses 125 rows: 346 rows / 128 standardized
identities overlap official train or test, versus 221 rows / 84 stored full
keys flagged in the source. All 346 are excluded from this experiment.
383 rows are not exact on both endpoints; 3 rows have nonpositive numeric
values. These exclusion reasons may overlap. README also documents 219 pairs
with differing cell lines, but the CSV does not carry both cell/exposure records
needed to verify compatibility row by row. README claims ChEMBL-only data while
8 rows identify PubChem as the source. The branch includes no reproducible
collection/merge script.

The snapshot builder normalizes record-pair ID formats, drops conflicting pairs,
deduplicates repeated record pairs, and accepts only positive exact numeric
observations disjoint from all competition identities. Its output has 104
records from 45 identities. Different measurements are not collapsed into a
single target: each identity has total external training weight .25 distributed
equally across its retained records. The individual records and source rows
remain traceable.

**This is a provisional numerical ablation, not a verified biological-label
dataset.** Endpoint types and experimental conditions may still differ.
Any apparent improvement cannot establish their compatibility or authorize
promotion. No source-paper audit was performed in this task.

For structural stress, external records sharing a nonempty scaffold with any
outer-validation row or current inner-validation row are also excluded from
that fit. Ordinary identity-group tests exclude all competition identities
globally. External fit indices are recorded for independent checks.

## Reproduce

```sh
PYTHONPATH=. python experiments/v9_similar_snapshot.py
PYTHONPATH=. python experiments/v9_backend_validation.py --freeze
PYTHONPATH=. python experiments/v9_backend_validation.py --protocol-sha256 PRINTED_HASH
python -m pytest -q tests/test_v9_model_backends.py tests/test_v9_similar_snapshot.py tests/test_nested_selection.py
```

Existing artifacts cannot be overwritten. The protocol pins source/data hashes,
all folds, backend parameters and dependency versions before fitting. Inner OOF,
outer OOF, weight choices and external training-row audits are saved separately.

## Results

Final executed protocol SHA256:
`6117a8480f64983017b5428b85e361e71a4ca62cb5a7caf12940a440e902a2d9`.
All scores below are full raw MSE over the 320 official rows. Group mean is
the arithmetic mean of identity/identity71/identity72; these repeats are correlated.

| Model / selection procedure | Primary identity | Group mean | Scaffold stress |
|---|---:|---:|---:|
| v4 baseline | 3,280,125 | 3,844,208 | 4,288,863 |
| ExtraTrees raw | 4,673,587 | 4,957,306 | 4,797,731 |
| ExtraTrees log | 4,238,119 | 4,254,633 | 4,283,742 |
| CatBoost raw | 4,367,255 | 4,367,042 | 4,351,201 |
| CatBoost log | 4,273,012 | 4,280,265 | 4,291,556 |
| Kernel ridge raw | 4,479,522 | 4,507,838 | 4,387,645 |
| Kernel ridge log | 4,279,399 | 4,282,778 | 4,294,906 |
| ExtraTrees log + provisional external | 4,228,805 | 4,254,080 | 4,276,072 |
| Inner-selected core single | 4,231,351 | 4,302,605 | 4,279,716 |
| Inner-selected core ensemble | 4,247,236 | 4,309,431 | 4,309,071 |
| Inner-selected exploratory single | 4,239,635 | 4,309,616 | 4,278,838 |
| Inner-selected exploratory ensemble | 4,239,635 | 4,304,202 | 4,309,068 |

The predeclared primary core ensemble loses to v4, and the confirmation rule
fails. Its group mean is 12.10% worse. Keep v4 as the accepted model; no new
submission or deployable full-data model was produced.

The external-data ablation changes ExtraTrees log group mean from 4,254,633.420
to 4,254,079.794 (-0.013%). It improves primary and repeat72 slightly but worsens
repeat71 (4,270,337 → 4,291,896). This is not a stable, practically substantial
improvement. It does not validate the external measurements' compatibility.

One competition test structure is rejected by the existing standardization
curation bounds. Training and external identities must all standardize; exact
SMILES overlap is also excluded for every competition row, including that test
row. Its rejection count is explicitly recorded. No rows of the training set
were removed, and no test labels were read.

Validation commands:

```sh
python -m pytest -q tests/test_v9_backend_validation.py tests/test_v9_similar_snapshot.py tests/test_v9_model_backends.py tests/test_nested_selection.py
PYTHONPATH=. python experiments/audit_v9_backends.py
```

The 99 targeted tests passed. Independent audit passed 1,438 checks: 48 metric
rows, 40 selections and 80 fitting partitions, source/artifact hashes, external
weights/overlap and exact v4 prediction parity with V8. This numerical isolation
audit is not a biological source-record audit.
