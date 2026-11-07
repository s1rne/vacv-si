# Symmetric pair tail weighting

Discovery46 frozen winner: `power0125`.

| Split | Model | Full raw MSE |
|---|---|---:|
| 46 | power0125 | 4253073.815 |
| 46 | power025 | 4436523.936 |
| 46 | power05 | 5030718.673 |
| 46 | v4-ridge-log | 4021684.009 |
| 47 | power0125 | 3962154.647 |
| 47 | v4-ridge-log | 3406357.889 |
| 48 | power0125 | 3938578.085 |
| 48 | v4-ridge-log | 3239105.619 |
| 49 | power0125 | 3747684.129 |
| 49 | v4-ridge-log | 3057714.173 |
| 50 | power0125 | 3973002.751 |
| 50 | v4-ridge-log | 3306347.271 |
| scaffold | power0125 | 4254360.684 |
| scaffold | v4-ridge-log | 4253853.162 |

Confirmation means: {"power0125": 3905354.903073011, "v4-ridge-log": 3252381.2381056855}; paired wins: 0/4.

Three predeclared powers: 0.125, 0.25, 0.5. Validation uses the unchanged v4 Ridge-log baseline on identical folds. Only mean-one symmetric pair weights ((1+y_i)*(1+y_j))**power vary; alpha=10, fingerprints, pairs, fallback and query anchors unchanged. Weights access outer-training labels only. All 320 labels retained, no cap/external data/forbidden concentration features. Repeated CV is not independent data, and local scores are not comparable numerically with public score. No submission generated or uploaded.
