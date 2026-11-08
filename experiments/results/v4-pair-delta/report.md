# Train-only molecular pair differences

Frozen discovery winner: `ridge-log`.

| Split | Model | Full raw MSE |
|---|---|---:|
| 33 | ridge-raw | 6717314.684 |
| 33 | ridge-log | 4166288.993 |
| 33 | et-raw | 8893822.453 |
| 33 | et-log | 4759715.497 |
| 33 | context-raw70 | 4580395.132 |
| 34 | ridge-log | 3132247.900 |
| 34 | context-raw70 | 3778834.646 |
| 35 | ridge-log | 4111628.932 |
| 35 | context-raw70 | 4061464.647 |
| scaffold | ridge-log | 4253853.162 |
| scaffold | context-raw70 | 4290081.636 |

Confirmation mean: 3621938.416; improves both paired: False.
4M local gate: False; 3M each upload gate: False.

All 320 labels retained. Same-context pairs selected using structures only inside each outer training fold; both orientations included. Morgan bit differences; Ridge alpha=10 or ET 400 trees leaf=3; raw/1000 or log1p deltas. Top-8 pair neighbors at similarity >=0.5, top-3 prediction anchors >=0.7, similarity^8 weighting; structural log-KNN fallback. Nonnegative final outputs, no upper cap. Identical allowed representations stay in their predeclared random folds; scaffold is a separate stress test. These are local CV results, not independent leaderboard evidence. No submission was generated or uploaded.
