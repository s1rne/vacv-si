# Bounded power regression, official v2

Six fixed configurations evaluated on random 5-fold discovery seed 9; frozen winner `sqrt50-log50` confirmed on seeds 10/11 and scaffold folds. All 320 labels, including 36,066.667, remain in fitting and raw MSE. No external data or forbidden fields used. Power transforms are `(1+y)^p-1` and log1p; context/structure ExtraTrees predictions average in transformed space, invert, then mix in raw space. Bounds come only from training targets. 800 trees, seeds 0/1/2, leaf 2, feature fraction 0.5.

| Split | Log reference MSE | Frozen power MSE |
|---|---:|---:|
| Discovery 9 | 4,242,875.83 | 4,224,893.65 |
| Confirmation 10 | 4,256,316.42 | 4,222,729.79 |
| Confirmation 11 | 4,279,849.95 | 4,275,528.37 |
| Confirmation mean | 4,268,083.18 | 4,249,129.08 |
| Scaffold | 4,262,616.93 | 4,237,887.88 |

Confirmation mean improves 0.444%, but neither confirmation reaches the user threshold of 3M; **not eligible for Kaggle submission**. Diagnostics excluding the largest label are secondary only and did not select the winner. No Kaggle submission made.

Run from root: `PYTHONPATH=. python experiments/v2_power.py` in the project RDKit/sklearn environment. The script refuses an existing nonempty results folder. It writes partial fold recovery files and complete OOFs; only complete OOFs belong in the checkpoint. `manifest.json` records configuration, official train hash and full submission hashes. Both CSVs checked for 597 rows, exact template order, finite nonnegative values. The log implementation was checked against the earlier log baseline on 25 training/10 query rows: maximum absolute difference 1.42e-14 from floating-point summation order.

Validation: `python -m pytest tests/test_power_regression.py tests/test_regression_v2.py -q` passed 14 tests. Initial experiment export failed because official training data has no index column; fixed to positional OOF row IDs and repeated the complete unchanged experiment. Partial OOF recovery is now saved after every fold.
