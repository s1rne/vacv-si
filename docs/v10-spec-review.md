# Review of the proposed improvements, 2026-11-08

Source proposal: `docs/improvement-spec-2026-11-08.md` in the separate main checkout.
The proposal is evaluated as research input, not adopted as instructions or facts.
Baseline is the submitted v4 predictor with recorded Public MSE 1,710,149.26680.
The authenticated Kaggle page was checked read-only on 2026-11-08: its recent
successful submission is `submission-pair-delta-v4.csv`, Public MSE
1,710,149.26680. The experiment generates no test predictions or submissions.

## Assessment fixed before outcomes

1. **Leaderboard probing: not implemented.** The quadratic scale identities
   follow from MSE, but fitting against public scores uses hidden evaluation
   answers as a training signal. It does not establish transfer to new data.
   The quoted public moments and hidden test values are not independently
   established here and are not inputs to any experiment. Existing no-upload
   instructions remain in force. Local nested calibration tests the useful
   general principle without probing public labels.
2. **Chiral features: justified isolated ablation.** Representation sensitivity
   is a valid general ML hypothesis. Test a chiral Morgan variant of the same
   baseline without changing its model parameters, adding new source labels,
   or assuming the expected answer of any test molecule. Do not globally mutate
   all historical features or cached embeddings. Keep non-stereochemical identity
   grouping so related stereoisomers cannot straddle the validation folds.
   This ablation does not test every suggested stereo descriptor or encoder.
3. **OOF scale and prediction cap: justified with nested assessment.** Fit the
   nonnegative scalar on inner OOF targets only, then score outer predictions.
   Evaluate an independent fixed prediction cap5000 and the scale-then-cap
   combination. The analytic scalar is optimal for unbounded squared loss;
   it is not necessarily optimal after clipping. Report these as fixed variants,
   not a jointly optimized clipped estimator. Never evaluate the calibration
   on its fitting OOF and call that an independent validation result.
4. **Training-target cap2500: not adopted.** It changes the learning objective
   and sacrifices representation of rare large outcomes. The proposed rationale
   relies on unknown test labels and prioritizes an outlier-excluded metric.
   All original targets remain in fitting and primary scoring; without-maximum
   metrics are diagnostics only. No claim that target clipping is universally
   invalid: it would require a justified target-distribution assumption here.

## Corrections to assumptions

The current official train has 320 rows, **three** targets above2500, maximum
36,066.667 and next-largest5000. Its NumPy linear-interpolated 99th percentile
is **2466.73385**, not5000. The chosen prediction cap5000 is therefore a fixed
hypothesis, not an estimated P99. The cited CV1.048M belongs to a different
pipeline and is not the paired baseline for our saved v4 model.

Bounds derived from public-score moments concern that public subset only;
they do not prove absence of large outcomes in the private test or future data.
Using forbidden competition columns as auxiliary training targets is also not
authorized by our existing constraints and is excluded.

## Validation contract

Reuse the materialized V8 identity, identity71, identity72 and identity/scaffold
union outer folds, each with its predefined three inner folds. Dataset and
source versions are frozen before fitting. For achiral/chiral representations,
report raw, inner-OOF scalar, fixed cap5000 and scalar-then-cap5000 separately.
No model selection on outer targets, no external labels, no test-row inference.
Compare every variant with v4 on the same full raw-MSE rows. A robust improvement
must improve all three identity schemes without worsening structural stress;
none of these historically reused splits is a fresh independent dataset.

This measures changes in local validation relative to the submitted predictor.
It cannot measure a new Kaggle score without a submission.


## Completed results

160 fits completed under protocol SHA256
`bce1c90cf77f18c405a1d46e25f76b8a886a31051473512d76137558f3feda04`.
All 320 original targets remain in fitting and primary scoring.

| Fixed variant | Identity | Identity71 | Identity72 | Scaffold | Mean identity MSE |
|---|---:|---:|---:|---:|---:|
| Submitted v4 baseline | 3,280,125 | 4,312,185 | 3,940,314 | 4,288,863 | 3,844,208 |
| v4 + nested scalar | 4,039,549 | 4,376,270 | 4,210,704 | 4,287,379 | 4,208,841 |
| Chiral representation | 4,241,962 | 4,290,448 | 4,256,029 | 4,283,845 | 4,262,813 |
| Chiral + nested scalar | 4,260,645 | 4,279,363 | 4,269,177 | 4,311,589 | 4,269,728 |

Prediction cap5000 changes no outer prediction, with or without calibration;
its four variants therefore reproduce the corresponding rows above. No variant
passes the predefined robustness criterion. Mean identity MSE is descriptive,
not an independent estimate or a selection criterion.

Chirality improves the diagnostic MSE excluding the maximum-target row in all
three identity schemes (205,962 -> 194,835; 253,508 -> 234,515;
230,486 -> 211,018), but full MSE worsens in two of three schemes. This is a
mixed signal, not proof that chirality is useless. It does not justify replacing
the submitted predictor. Scalar calibration worsens both full and diagnostic
identity MSE in all three schemes. The large single-row contribution makes
these small-data estimates unstable; deleting that row would change the task.

The baseline OOF predictions exactly reproduce saved V8 predictions in all four
schemes (maximum absolute difference 0). This is a paired local comparison to
the algorithm used for the last submission; it is not a new Public score.
No candidate is promoted, no Kaggle upload is performed.

## Verification and reproduction

Executed successfully (11 tests passed):

```sh
PYTHONPATH=. DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs /Users/mike/Projects/vacv-si-novel/.venv/bin/python -m pytest -q tests/test_v10_spec_validation.py tests/test_v4_pair_delta.py tests/test_v8_nested_validation.py
```

The run uses `experiments/v10_spec_validation.py --protocol-sha256` with the
hash above after a separate `--freeze` step. Both protocol and output paths
refuse overwriting: reproducing requires a separate checkout/output location
with matching frozen sources, input data and dependency versions.

Saved evidence: `experiments/results/v10-spec-validation/metrics.json`,
40 inner OOF CSVs, 8 outer OOF CSVs, calibration audit, baseline parity,
protocol copy and artifact manifest. Synthetic tests cover held-out-label
invariance, scalar optimization, prediction-only clipping and stereoisomer
representation discrimination.
