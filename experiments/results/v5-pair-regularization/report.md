# Pair regularization ablation

Discovery41 frozen winner: `alpha100`.

| Split | Model | Full raw MSE |
|---|---|---:|
| 41 | alpha1 | 4823590.024 |
| 41 | alpha3 | 4823446.494 |
| 41 | alpha30 | 3584288.529 |
| 41 | alpha100 | 3410239.888 |
| 41 | degree-alpha10 | 4227451.789 |
| 41 | v4-ridge-log | 4210499.382 |
| 42 | alpha100 | 3530052.655 |
| 42 | v4-ridge-log | 3347464.347 |
| 43 | alpha100 | 4143305.579 |
| 43 | v4-ridge-log | 4098286.076 |
| 44 | alpha100 | 3182442.585 |
| 44 | v4-ridge-log | 3012754.903 |
| 45 | alpha100 | 4266845.229 |
| 45 | v4-ridge-log | 4266377.334 |
| scaffold | alpha100 | 4266019.526 |
| scaffold | v4-ridge-log | 4253853.162 |

Confirmation means: {"alpha100": 3780661.5119507676, "v4-ridge-log": 3681220.665029743}; paired wins: 0/4.

Five predeclared profiles. Validation uses the unchanged v4 Ridge-log baseline on identical folds. Only Ridge alpha and mean-one degree weighting vary; fingerprints, pairs, fallback and query anchors unchanged. All 320 labels retained, no cap/external data/forbidden concentration features. Repeated CV is not independent data, and local scores are not comparable numerically with public score. No submission generated or uploaded.
