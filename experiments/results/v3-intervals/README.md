# Record-paired interval regression, official v2

Three predeclared profiles use the same binary Morgan Tanimoto kernel and
RKHS ridge penalty alpha=1: competition-only, external exact observations,
and external exact plus one-sided bounds. External weight is fixed at 0.25.
Squared distance to each log1p-SI interval is zero inside the permitted
interval. Lower and upper bounds are never converted into exact labels.
Predictions invert log1p with no upper clipping; negative predictions become
zero. Every competition target, including 36,066.667, is retained.

The approved technical snapshot is `../v3-source-pairs/eligible.csv`, SHA256
`9ead9fc63139b88352584a39673f3458beaf1abb8c3ddd776c0de7d1729a2e23`.
It contains 170 distinct molecules: 47 exact, 41 lower and 82 upper bounds.
The experiment independently rechecked all train/test InChIKey14 and exact
SMILES exclusions, invalid structures and duplicate keys: zero excluded
rows. This is the reconstructed record-pair snapshot, not the rejected GLM
`measured_clean.csv` aggregation. Endpoint selection and same-cell matching
are documented by the source builder; full assay-protocol equivalence is
not independently verified for every row, and missing exposure metadata is
a remaining limitation. The high-SI record is retained without a target cap.

Seed21 selected `external-intervals`; the choice and all hyperparameters
remained unchanged on confirmation22/23 and scaffold. All three fixed
profiles are retained as paired ablations; no confirmation-based reselection.

| Split | Competition only | External exact | Exact + intervals |
|---|---:|---:|---:|
| Discovery21 | 4,275,406.71 | 4,273,986.75 | 4,273,734.51 |
| Confirmation22 | 4,275,218.50 | 4,273,892.12 | 4,273,672.60 |
| Confirmation23 | 4,275,143.25 | 4,273,813.36 | 4,273,870.92 |
| Scaffold | 4,288,752.68 | 4,287,460.46 | 4,287,384.24 |

Full raw-MSE only. The external data slightly improve the no-external model,
but bounds do not consistently improve exact-only training: confirmation23
is slightly worse. The maximum training target is still predicted as only
63.49 and 71.21 on the two confirmations. Thus this experiment does not
address the dominant tail error and does **not** meet the <=3M gate on
either confirmation. No candidate CSV was created and no Kaggle upload made.

Validation: six interval-regression tests passed (analytic versus numerical
gradient, bound direction, feasible bounds exerting no attraction, exact-case
kernel-ridge equivalence, invalid intervals). All four OOF files contain the
320 official targets exactly once, five folds, finite nonnegative predictions.

Reproduce from the repository root into a fresh output directory:

```sh
PYTHONPATH=. python experiments/v3_intervals.py \
  --approved-external experiments/results/v3-source-pairs/eligible.csv \
  --out /path/to/fresh-results
python -m pytest -q tests/test_interval_regression.py
```
