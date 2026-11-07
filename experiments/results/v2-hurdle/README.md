# Hurdle raw-SI experiment

Two fixed configurations split training labels into exact zero/nonzero or below100/
at-least100. Each combines an unweighted ExtraTrees group classifier with separate
raw-SI regressors in both groups: prediction is the probability-weighted sum of
their conditional means. Both seeds use 400 trees, minimum leaf2, max features0.5.
Groups with fewer than four samples use their training group mean; absent groups
use the full training mean. Class probabilities are not balanced or reweighted.

Inputs: fold-local ContextEncoder plus Morgan/MACCS fingerprints. All official-v2
training labels remain present, including zeros and SI36066.667. No sentinel
semantics, competition IC50/CC50, test targets, external lookup, or manual labels.

| Profile | Discovery9 MSE | Confirmation10 | Confirmation11 | Scaffold |
|---|---:|---:|---:|---:|
| cutoff0 | 5,827,365 | 6,184,478 | 4,992,191 | 4,809,901 |
| cutoff100 | 7,344,817 | not selected | not selected | not selected |

Discovery winner cutoff0 was fixed before confirmation. Confirmation mean5,588,334
is poor: conditional raw regression concentrates the influence of the extreme
label in the smaller positive group, producing false positive predictions up to
13.6k. Cutoff100 is worse, reaching15.8k. Do not submit these candidates to Kaggle.
The saved winner CSV is for reproducibility only.

Validation: constant-zero and single-class cases, exact100 threshold boundary,
constant36k preservation, complete320-row OOF across four splits, finite
nonnegative outputs, and597 test IDs matching sample-submit schema and ordering.

Run from repository root with an empty output directory:

```sh
DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs PYTHONPATH=. /Users/mike/Projects/vacv-si-novel/.venv/bin/python experiments/v2_hurdle.py
```
