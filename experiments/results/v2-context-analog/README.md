# Context-gated analog expert

Hypothesis: raw target transfer is useful only for structurally similar molecules
with the same assay context. This tests a structural change to the predictor,
rather than selecting another hyperparameter on the previous confirmation folds.

Three fixed profiles were declared before the fresh seed12 discovery evaluation:

1. Raw mean of up to3 same-context neighbours with Tanimoto at least0.70.
2. The same raw expert with threshold0.85.
3. Log1p-target expert at threshold0.70, isolating the effect of the target scale.

Both strain and Culture must match after whitespace/case normalization; missing
context does not match. Only neighbours above the stated threshold are eligible.
Weights are Tanimoto to power8. If no eligible neighbour exists, prediction falls
back to structure-only top8 log-KNN. No test labels, external tables, manual edits,
special-case compounds, or target clipping are used. All320labels including36k
remain in the raw-MSE comparison.

| Profile | Discovery12 | Confirmation13 | Confirmation14 | Scaffold |
|---|---:|---:|---:|---:|
| raw70 | **3,981,656** | 4,167,764 | 3,886,398 | 4,290,082 |
| raw85 | 4,249,975 | not selected | not selected | not selected |
| log70 | 4,095,759 | not selected | not selected | not selected |
| Structure-log fallback control | 4,249,974 | 4,257,969 | 4,261,803 | 4,282,619 |

Raw70 was frozen after discovery12. Its confirmation mean4,027,081 improves on
the paired fallback control4,259,886 by5.465%. However, the scaffold score is
slightly worse. Tail coverage depends on having close, same-context analogues:
leave-one-out raw70 covers5of7 labels at least1000, including the maximum; raw85
covers none. The raw70 maximum-label prediction is1274/2325 on confirmation13/14,
but only20 on scaffold. This limits extrapolation to new chemical series.

Feature-only coverage:

| Query set | Matching known context | Same-context sim>=0.70 | Same-context sim>=0.85 |
|---|---:|---:|---:|
| Train leave-one-out,320rows | 309 | 134 | 20 |
| Test,597rows | 588 | 277 | 64 |

The fallback-only control was computed after fixed winner selection as a paired
diagnostic; it did not change any expert predictions or config selection. Exact
production `RegressionV2(et-blend-log)` comparison is stored separately under
`production-baseline`, with a cacheCSV per fold and combined OOFs.

Exact production ET confirmation13/14 scores are4,261,720 and4,254,731 (mean
4,258,226). Raw70 reduces the mean by5.43%, entirely through improved prediction
of the maximum label. Its maximum-label contribution decreases by269,800/498,075,
while non-tail errors add178,758/134,669 MSE on the two confirmation splits.
Detailed components are saved in `production-error-decomposition.csv`.

**Not eligible for Kaggle submission:** neither confirmation score meets the new
requirement of full raw MSE<=3,000,000 on both independent splits.

`train-tail-neighbors.csv` and `train-tail-coverage.json` document the limitation:
the36k molecule has only two same-context neighbours above0.70 similarity, with
targets2324.915 and223 (both similarity0.7174). Any convex average restricted to
these neighbours predicts at most2324.915, producing at least3,557,831 full-MSE
contribution from this one held-out row, even if every other prediction is exact.
This expert family therefore cannot meet3M by tuning its neighbour averaging.
Moreover, one5000-target molecule's nearest same-context neighbour (similarity0.8)
has SI11.95, showing an activity cliff; another1811.594-target molecule has no
same-context training neighbours. No labels were modified to address these cases.

More generally, the second-largest official training target is5000. Any train-only
predictor bounded by its fold-training target range has a lower bound of
3,016,055.62 full MSE from the held-out maximum alone. Thus the new3M criterion
requires genuine extrapolation or independent permitted external training evidence;
ordinary tree averages and convex KNN cannot pass even with ideal predictions for
all other rows. The maximum-label prediction must exceed5082.80 before accounting
for any other errors. See `nonextrapolating-lower-bound.json`.

Production scaffold MSE is4,262,617, versus4,290,082 for the gated expert. All
production OOF row IDs, targets and fold assignments match the expert runs.

Validation passed: complete320-row OOFs and five folds, exact597-ID submission
schema and order, finite nonnegative predictions, exact full-model re-prediction
of submissionCSV, identity-neighbour recovery including36k, rejection of unknown
context matches. Candidate SHA256:
`dbe5abc0642571c6641809af1f0f6e53551a8f627d9a32a7269486f5e5971b7b`.

Run from repository root with an empty experiment output directory:

```sh
DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs PYTHONPATH=. /Users/mike/Projects/vacv-si-novel/.venv/bin/python experiments/v2_context_analog.py
```

Append `--production-baseline` to run/resume the paired production baseline;
this writes only its separate artifact subdirectory.
