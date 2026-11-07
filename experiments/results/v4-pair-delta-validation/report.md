# Frozen pair-delta validation

Fixed Ridge-log; fixed seeds 13, 14, 39, 40. Repeated CV on the same 320 molecules, not independent data. No retuning or seed selection.

| Seed | Ridge-log full raw MSE | Context-KNN full raw MSE | Improvement |
|---|---:|---:|---:|
| 13 | 4094047.616 | 4167763.737 | 1.769% |
| 14 | 2178510.793 | 3886398.467 | 43.945% |
| 39 | 4023416.406 | 4117446.047 | 2.284% |
| 40 | 3524386.626 | 4010315.298 | 12.117% |

All four means: pair=3455090.360, baseline=4045480.887.
All labels retained, including maximum. Forbidden concentration columns and external data excluded. Complete OOF predictions and fold membership saved. No submission CSV generated or uploaded.
