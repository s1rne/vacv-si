# Pair-delta molecular representation experiment

Frozen winner on seed41: morgan-r2-r3-bit. Paired control is v4 Ridge-log, the model family associated with the user's reported ~1.710M public score; local CV is not that public score.

| Split | Representation | Full raw MSE |
|---|---|---:|
| 41 | morgan-r3-bit | 4209568.266 |
| 41 | morgan-r2-logcount | 4219004.578 |
| 41 | morgan-r2-r3-bit | 4202503.819 |
| 41 | v4-ridge-log-r2-bit | 4210499.382 |
| 42 | morgan-r2-r3-bit | 4075008.610 |
| 42 | v4-ridge-log-r2-bit | 3347464.347 |
| 43 | morgan-r2-r3-bit | 4197494.632 |
| 43 | v4-ridge-log-r2-bit | 4098286.076 |
| 44 | morgan-r2-r3-bit | 3994892.767 |
| 44 | v4-ridge-log-r2-bit | 3012754.903 |
| 45 | morgan-r2-r3-bit | 4275190.737 |
| 45 | v4-ridge-log-r2-bit | 4266377.334 |
| scaffold | morgan-r2-r3-bit | 4260036.858 |
| scaffold | v4-ridge-log-r2-bit | 4253853.162 |

Confirmation mean: 4135646.686; paired v4 mean: 3681220.665; relative change: 12.344%.
Improves every confirmation: False. Each confirmation <=3M: False.

Reject this representation change: the frozen discovery winner loses on all four confirmation seeds, with mean MSE 12.344% worse. On seeds42/44 the maximum-SI row is predicted at about978/1004 versus4978/5816 for the unchanged v4 baseline. Larger-radius features reduce useful extrapolation in these folds. Scaffold also worsens. This does not justify replacing the public-score incumbent. Two focused tests passed, covering deterministic structure-only fingerprints and prediction invariance when query labels are changed to an extreme value.

Only the delta vector changes: radius3 2048-bit, radius2 log1p hashed counts2048, or concatenated radius2/radius3 binary vectors with each block scaled by 1/sqrt(2). All use existing v4 Ridge alpha10, no intercept, log1p target deltas, antisymmetry, same training pairs, top3 anchors, original radius2 similarity thresholds and fallback. No hyperparameter search. Every label remains in full raw MSE, without target cap/removal. All learned pair labels restricted to outer train. No external data, forbidden fields, candidate CSV or uploads. Scaffold is a stress test.
