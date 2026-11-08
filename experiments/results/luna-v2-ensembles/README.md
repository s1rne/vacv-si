# Luna v2 OOF ensemble audit

All comparisons use identical positional rows, full targets and fold IDs across the saved source OOFs. The paired ET reference is `v2-power` `log` (the same three-seed ExtraTrees context/structure pipeline); no seed-6/7/8 baseline is mixed into these comparisons.

Seed 9 selected `et_tweedie_50`. Frozen confirmation MSE: seed 10 4223317.827; seed 11 4278211.294; scaffold 4266479.580.
Paired full raw-MSE comparison (all 320 rows):

| Split | ET log baseline | Frozen blend | Delta | Change |
|---|---:|---:|---:|---:|
| 9 | 4,242,875.828 | 4,185,366.322 | -57,509.507 | -1.355% |
| 10 | 4,256,316.416 | 4,223,317.827 | -32,998.590 | -0.775% |
| 11 | 4,279,849.945 | 4,278,211.294 | -1,638.651 | -0.038% |
| scaffold | 4,262,616.934 | 4,266,479.580 | +3,862.646 | +0.091% |
3,000,000 confirmation gate: FAIL. No test candidate CSV was produced and no Kaggle upload was made.

All 320 rows and the 36,066.667 target are retained; only SMILES, Title, strain and Culture are used as features by the source experiments. The ensemble is a convex average of saved predictions, with weights fitted on seed 9 only. No test labels or public scores were used. The train-only prediction cap implies a 3.016M lower bound from the held-out maximum alone; these convex blends therefore cannot satisfy the 3M gate.

No candidate CSV was generated because the frozen candidate fails the gate. No Kaggle upload was made.

Reproduce from the repository root with the existing environment:

```sh
DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs PYTHONPATH=. /Users/mike/Projects/vacv-si-novel/.venv/bin/python experiments/luna_v2_ensembles.py
```

See `metrics.json`, `report.json`, and the four complete OOF files for source hashes and predictions.
