# Query-local pair Ridge

Discovery51 frozen winner: `half-local4`.

| Split | Model | Full raw MSE |
|---|---|---:|
| 51 | local4 | 4149228.015 |
| 51 | local8 | 4128541.431 |
| 51 | half-local4 | 4118950.587 |
| 51 | v4-ridge-log | 4089231.368 |
| 52 | half-local4 | 3977560.979 |
| 52 | v4-ridge-log | 3886034.771 |
| 53 | half-local4 | 3755927.112 |
| 53 | v4-ridge-log | 3560667.962 |
| 54 | half-local4 | 3093515.258 |
| 54 | v4-ridge-log | 2746470.846 |
| 55 | half-local4 | 4193529.910 |
| 55 | v4-ridge-log | 4187238.661 |
| scaffold | half-local4 | 4253763.867 |
| scaffold | v4-ridge-log | 4253853.162 |

Confirmation means: {"half-local4": 3755133.314621242, "v4-ridge-log": 3595103.0599298924}; paired wins: 0/4.

Three predeclared profiles: local power4, local power8, 50/50 local4 plus v4. Validation uses the unchanged v4 Ridge-log baseline on identical folds. Per-query mean-one pair weights: 0.1+0.9*max(sim(query,a),sim(query,b))**power. Query labels never used. Alpha=10; fingerprints, pair selection and query anchors unchanged. Ridge is refitted locally using outer-training pairs only. All 320 labels retained, no cap/external data/forbidden concentration features. Repeated CV is not independent data, and local scores are not comparable numerically with public score. No submission generated or uploaded.
