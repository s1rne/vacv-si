# Background-aware pair trees

Discovery56 frozen winner: `leaf5`.

| Split | Model | Full raw MSE |
|---|---|---:|
| 56 | leaf2 | 4321953.750 |
| 56 | leaf5 | 3951090.712 |
| 56 | v4-ridge-log | 4048951.271 |
| 57 | leaf5 | 3980766.674 |
| 57 | v4-ridge-log | 3670751.986 |
| 58 | leaf5 | 4145523.970 |
| 58 | v4-ridge-log | 4085042.386 |
| 59 | leaf5 | 3795776.025 |
| 59 | v4-ridge-log | 2573888.892 |
| 60 | leaf5 | 3761887.759 |
| 60 | v4-ridge-log | 3312471.624 |
| scaffold | leaf5 | 4254461.899 |
| scaffold | v4-ridge-log | 4253853.162 |

Confirmation means: {"leaf5": 3920988.60701443, "v4-ridge-log": 3410538.722002673}; paired wins: 0/4.

Two predeclared ExtraTrees profiles: 400 trees, max_features=0.5, min_samples_leaf=2 or 5. Validation uses the unchanged v4 Ridge-log baseline on identical folds. Pair features concatenate fingerprint difference and mean; predictions antisymmetrized while retaining the mean background. Pair selection, fallback and query anchors unchanged. Custom PADRE-inspired variant, not a reproduction of its all-pairs training protocol. Source: https://doi.org/10.1021/acs.jcim.1c00670 . All 320 labels retained, no cap/external data/forbidden concentration features. Repeated CV is not independent data, and local scores are not comparable numerically with public score. No submission generated or uploaded.
