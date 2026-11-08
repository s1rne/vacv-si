# Official-v2 molecular analog checkpoint

Four fixed configurations were declared before execution. Discovery uses shuffled
five-fold seed 9; confirmation uses seeds 10 and 11; scaffold grouping is a stress
test. All 320 targets, including SI 36066.667, are retained in every full-MSE
comparison. The diagnostic without the maximum target is not a selection metric.

| Model | Discovery MSE | Confirmation mean MSE | Scaffold MSE |
|---|---:|---:|---:|
| KNN8 raw | 4,833,400 | 4,830,350 | 5,489,189 |
| KNN8 log | 4,243,853 | 4,260,021 | 4,282,619 |
| KNN8 log + external | 4,244,257 | 4,259,044 | 4,280,612 |
| Local KRR24 raw | 4,574,769 | 4,661,373 | 4,459,590 |

Features are binary radius-2 Morgan fingerprints, 2048 bits. KNN averages the
eight nearest target values with Tanimoto similarity raised to power 8. The log
variant averages log1p targets, then applies expm1. External samples receive a
0.25 weight after neighbour selection. Local KRR fits the 24 nearest molecules,
Tanimoto-squared kernel, ridge 0.5, centered on their median target.

External data: measured ChEMBL rows only, median SI per standardized InChIKey14;
all official train and test keys excluded before fitting (316 external molecules,
zero overlap). No competition IC50/CC50 fields, missingness indicators, matched
evidence features, test targets, manual edits, or targeted external answer lookup.

Raw analog models overpredict low-target neighbours of the 36k training molecule:
random-fold KNN maxima reach 7–9k. Log models sharply reduce this error. External
augmentation yields only small confirmation gains and slightly worsens discovery;
there is no evidence here warranting a separate Kaggle submission. Comparisons to
ET on different fold seeds are not paired evidence of improvement.

`oof-*.csv` includes every row, target, fold and all four predictions.
`submission-*.csv` includes all 597 test rows in the sample submission order.
`metrics.json` records full MSE, tail/non-tail diagnostics and prediction ranges;
`manifest.json` records the source hash and leakage exclusion audit.

Run from repository root (output directory must not already contain artifacts):

```sh
DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs PYTHONPATH=. /Users/mike/Projects/vacv-si-novel/.venv/bin/python experiments/v2_analog.py
```

Validation passed for OOF row coverage, five-fold coverage, retention of the full
tail, finite nonnegative predictions, exact sample-submission row order, zero
external overlap, and exact nearest-neighbour recovery on an identity toy case.
