# Exact/interval kernel regression on raw and square-root SI

Six predeclared discovery configurations: raw SI/1000 or sqrt(SI), each with
competition-only training, added external exact observations, or added exact
and one-sided bounds. Same Tanimoto kernel, alpha=1, external weight=0.25.
The transformations preserve every label and every interval, including the
36,066.667 official target and the independently audited high external SI.
No upper clipping of targets or predictions. The only prediction floor is
zero; raw latent predictions multiply by 1000 and sqrt predictions square.

The frozen source snapshot contains 170 disjoint molecules (47 exact, 41 lower
and 82 upper bounds). Its SHA256 is
`9ead9fc63139b88352584a39673f3458beaf1abb8c3ddd776c0de7d1729a2e23`;
the runner checks that exact hash and independently rechecks competition
overlap, invalid structures and duplicate external keys (zero exclusions).
Same-paper/known-cell record matching does not prove identical full assay
protocols for every pair; missing durations remain a source limitation.

Discovery27 selected sqrt/external-intervals. Raw-scale external training
was worse: raw baseline4.755M, raw+exact5.316M, raw+intervals5.758M. No raw
profiles were reselected or tuned after discovery. Confirmation compares
only the frozen winner with its matching sqrt competition-only baseline.

| Full raw-MSE | Matching baseline | Frozen winner |
|---|---:|---:|
| Discovery27 | 4,255,274.70 | 4,254,391.32 |
| Confirmation28 | 4,237,192.79 | 4,230,479.87 |
| Confirmation29 | 4,246,223.57 | 4,236,306.24 |
| Confirmation mean | 4,241,708.18 | 4,233,393.06 |
| Scaffold | 4,270,486.99 | 4,251,931.85 |

The improvement is small (~0.196% confirmation mean). Predictions for the
held-out 36,066.667 target remain 176.55/144.03. Both confirmations fail the
<=3M gate. No candidate CSV was generated and no Kaggle upload occurred.

Validation: 10 tests passed (four scale tests and six interval-regression
tests), including round-trip preservation of zero, 36k, high external labels
and infinite bounds. All four saved OOF files contain the complete 320 labels
and five folds, with finite nonnegative predictions; see verification.json.

Reproduce from the repository root with the approved frozen snapshot:

```sh
PYTHONPATH=. python experiments/v3_interval_scales.py --out /path/to/fresh-results
python -m pytest -q tests/test_v3_interval_scales.py tests/test_interval_regression.py
```
