# V5 continuation after public1.710M

Reference: unchanged v4 Ridge-log, the model submitted with public score
1,710,149.26680. No v5 predictions have been submitted. No external GLM
snapshot is incorporated in this checkpoint.

## Shared confirmation folds42–45

Each family freezes a winner on discovery41. Lower full rawMSE is better.

| Frozen family | Mean fullMSE | Paired v4 | Paired wins |
|---|---:|---:|---:|
| Ridge alpha100 | 3,780,662 | 3,681,221 | 0/4 |
| Bayesian molecule bagging, raw mean | 3,696,436 | 3,681,221 | 2/4 |
| Radius2+3 delta fingerprints | 4,135,647 | 3,681,221 | 0/4 |
| Predeclared equal ensemble, including v4 | 3,812,068 | 3,681,221 | 0/4 |

Bagging achieves2,962,105 on seed44 but regresses on42/45. Its scaffold score
4,248,045 is slightly better than v4's4,253,853. A single favorable split is
not sufficient to promote it. The other families also worsen scaffold.
The equal ensemble was specified in `v5-protocol.md` before confirmation
results were read, and its weights were not subsequently optimized.

## Separate targeted hypotheses

After the first batch, two independent hypotheses use discovery46 and
confirmations47–50. These different seeds must not be compared directly with
the means above; each has an unchanged same-fold v4 control.

Symmetric tail weights on training pairs, with powers0.125/0.25/0.5, select
0.125. Confirmation mean3,905,355 versus3,252,381, losing all four. Both tail
and ordinary-row errors worsen. Training weights depend only on outer-train
labels. This family is rejected and is not added to the equal ensemble.

## Identity stress test

The frozen v4 model is also evaluated by GroupKFold5 using standardized
InChIKey14 identities. There are311 identities, including9 repeated groups
covering18 rows; no standardization failures. Replicate identities cannot
cross folds. FullMSE3,280,125, maximum-target prediction4,698.88. This is an
additional stress test, not a selected hyperparameter or independent dataset.

## Interpretation

The working model's useful extrapolation is easily suppressed by apparently
reasonable regularization and representation changes. Its rare high-SI errors
dominate rawMSE. Do not replace the submitted v4 with a more complex family
merely because discovery or one confirmation looks better. All320 targets,
including36,066.667, are retained throughout. Concentration columns, external
label substitutions and uploads are absent. Repeated CV reuses the same data.

The softer anchor thresholds0.55/0.60/0.65 select0.60 on discovery46.
Confirmations47–50 average3,529,141 versus3,252,381 (+8.51% worse), losing all
four. Scaffold improves to4,206,799, but the random-CV regression remains.
Additional weaker anchors dilute useful extrapolation. This family is also
rejected; the original0.70 threshold is preserved.

Scripts: `experiments/v5_*.py`; OOF, manifests and reports under the matching
`experiments/results/v5-*/` directories. Run with `PYTHONPATH=. python` in the
existing scientific environment; scripts refuse to overwrite their results.

Verification: full `python -m pytest -q` passed261 tests. The parent separately
recomputed83 reported model/split MSEs from OOF, checked all320 ordered labels,
five-fold coverage, finite nonnegative predictions, and recorded source hashes.
