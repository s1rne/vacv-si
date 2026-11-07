# Distributional log-location/variance correction

Three predeclared profiles: paired log1p baseline and two approximate lognormal conditional-mean corrections. One ExtraTrees mean model uses concatenated fold-local context and structural features (400 trees, leaf2, max_features0.5, seed0). This is a paired experimental baseline, not the larger production ET ensemble.

Within each outer training fold, three inner folds generate held-out log1p residuals. Their squared values train a conditional second-moment model (400 trees, leaf5); its prediction is mixed with the global inner-OOF residual second moment. Fixed conditional weights are0.25/0.75. Predictions are exp(mu + moment/2)-1. No outer validation labels, external labels, forbidden columns or test targets enter fitting. All320 targets, including36,066.667, remain intact.

Important limitation: the residual second moment is a variance proxy assuming zero conditional residual mean. 1+SI is not proven lognormal (zeros included), and model error is not distinguished from aleatoric variance. These are uncalibrated distribution assumptions, not calibrated uncertainty estimates.

| Split | Model | Full raw MSE | Held-out maximum target prediction |
|---|---|---:|---:|
| 24 | log-baseline | 4,256,966.62 | 36.04 |
| 24 | variance25 | 5,214,357.81 | 160.46 |
| 24 | variance75 | 2,741,044,303.07 | 168.99 |
| 25 | log-baseline | 4,294,282.62 | 39.63 |
| 26 | log-baseline | 4,240,943.06 | 59.09 |
| scaffold | log-baseline | 4,271,297.44 | 44.04 |

Discovery24 selected `log-baseline`; confirmation25/26 and scaffold were not used to retune candidates. Gate failed; no candidate CSV and no Kaggle upload. Runtime128.43s.

The variance corrections hurt discovery: maximum false predictions15,702/927,166 while the36k observation is still predicted only160/169. This is a negative modeling result, not an improvement. Further weights or target clipping were not tried after seeing the validation results.

Checks:2 tests passed (inner residuals exclude their own labels; query target/forbidden fields cannot alter predictions). Independently checked320 rows, all5 folds, preserved targets, finite/nonnegative predictions and recomputed full raw MSE for every stored OOF. See verification.json.

Run from root with empty output directory: `PYTHONPATH=. python experiments/v3_distributional.py`. Hashes and fixed profile definitions are in manifest.json.
