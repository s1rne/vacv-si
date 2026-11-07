# Frozen nested evaluation

This experiment compares existing predictors with the submitted v4 baseline.
It introduces no external labels, creates no test predictions and never uploads.
The historically reused 320-row dataset is not a fresh independent holdout.

## Protocol fixed before execution

- Official `train_v2`, all 320 targets retained, full raw MSE. No row removal or target caps.
- Baseline: existing `v4` Ridge-log, unchanged.
- Four candidates: v4, existing context-raw70, existing pretrained embedding-pair
  and mixed-pair representations. The latter use the existing PCA32/scaler
  settings with seed0 and scales8/4. They are refitted on each fitting subset.
- Primary: deterministic identity GroupKFold5, matching the earlier identity
  stress test. Secondary: identity GroupKFold5 shuffled with seeds71 and72;
  structural stress: GroupKFold5 over connected components of shared identity
  OR shared nonempty scaffold. Empty scaffolds alone do not merge unrelated rows.
- Every outer train has its own shuffled GroupKFold3, seed1700+outer-fold-index,
  using the corresponding identity or connected-component groups. Both levels
  prohibit identity overlap; the structural scheme also prohibits scaffold overlap.
- Single-model selection minimizes pooled inner OOF MSE.
- Ensemble selection minimizes pooled inner OOF MSE across the 35 convex weight
  vectors with weights in quarters summing to one. Exact ties use descending
  lexicographic order, starting with all weight on v4. No outer-fitted weights.
- Primary endpoint: selected-ensemble pooled outer OOF MSE against v4, paired
  on the same rows. Selected-single and fixed candidates are secondary diagnostics.
  They cannot replace the primary endpoint after looking at results.
- Confirmation requires improvement on primary and both grouped repeats,
  with no worsening on structural stress. Regardless of results, no Kaggle upload.
- Outer labels are unavailable to fitting callbacks. Each callback receives only
  its fitting targets; legacy index-based predictors receive NaN elsewhere.

The protocol JSON contains materialized outer/inner indices, weight-grid order,
dependency versions, data hashes and relevant source hashes. Freeze first, then
pass the printed SHA256 to run. Existing artifacts cannot be overwritten.

```sh
PYTHONPATH=. python experiments/v8_nested_validation.py --freeze
PYTHONPATH=. python experiments/v8_nested_validation.py --protocol-sha256 PRINTED_SHA256
python -m pytest -q tests/test_nested_selection.py tests/test_v8_nested_validation.py
```

Each run saves inner OOF, selected weights, all outer OOF, full metrics and artifact
hashes in `experiments/results/v8-nested-validation/`. Models and weights are
selected anew within each outer training set; there is no single deployable
full-data ensemble from this validation-only experiment.

Grouped repeats reuse the same rows and are correlated. They are sensitivity
checks, not independent datasets or a calibrated forecast of the public score.
The frozen embedding cache is hash-checked; the protocol does not prove absence
of overlap with its original unsupervised pretraining corpus.

## Completed result

Frozen protocol SHA256:
`f424d1b593a2d182877d8ba181fcb537186e26e470c6c723207b8ae0cf32dc96`.

All cells below are full raw MSE over all 320 rows. Lower is better.

| Outer scheme | v4 baseline | Inner-selected single | Inner-selected blend | Blend vs v4 |
|---|---:|---:|---:|---:|
| Identity primary | 3,280,125.269 | 2,301,227.958 | 2,919,565.546 | -10.99% |
| Identity shuffled71 | 4,312,185.177 | 4,316,415.247 | 4,320,604.285 | +0.20% |
| Identity shuffled72 | 3,940,314.211 | 4,118,649.688 | 3,975,611.959 | +0.90% |
| Identity/scaffold union | 4,288,863.452 | 4,288,833.546 | 4,288,690.923 | -0.004% |

The primary endpoint improves, but the predeclared confirmation rule fails:
the ensemble loses both grouped repeats. The best accepted model remains v4.
The primary single-selection improvement of 29.84% is a secondary result and
does not replace the predeclared ensemble endpoint. On primary folds all five
inner selections choose embedding-pair; the blend adds v4/context weight in
one fold. On repeats, selected models and weights vary. Inner optimization
does not guarantee improvement on the outer fold, even with correct isolation.

These structural-stress folds use identity/scaffold unions and GroupKFold;
they differ from earlier scaffold-only tests. Only paired results in this run
are directly comparable. Primary v4 predictions and folds were independently
matched to the earlier `v5-identity-stress` artifact to numerical tolerance.

Validation: 53 relevant tests passed, covering target isolation, malformed
partitions, deterministic selection, grouped splits and train-only PCA.
Independent artifact audit recomputed all 20 selections, 24 outer metrics and
80 outer/inner group partitions, verified all frozen hashes, and reconstructed
outer selected predictions from the stored base predictions and inner weights.

```sh
PYTHONPATH=. python experiments/audit_v8_nested.py
python -m pytest -q tests/test_nested_selection.py tests/test_v8_nested_validation.py tests/test_v4_pair_delta.py tests/test_dataset_v2.py
```

No test-set predictions, new external-data training, Kaggle submission or
full-data ensemble fitting occurred. Pre-existing Terra/v7 files were untouched.
