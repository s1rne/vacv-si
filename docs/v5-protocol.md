# V5 stability protocol

The reference is the submitted frozen v4 Ridge-log (public MSE1,710,149.26680).
The user authorized continued local research, not another upload.

Three separate families select their winner on seed41 only:

- Ridge alpha1/3/30/100 and degree-normalized training-pair weights at alpha10.
- Sixteen molecule-weighted Bayesian-bootstrap fits: raw/log averaging and
  fixed half raw-bag / half v4 blending.
- Radius3 binary, radius2 log-count, and normalized concatenated radius2/3
  delta representations; original v4 similarity still selects pairs/anchors.

All frozen winners are evaluated on the SAME seeds42/43/44/45 and scaffold,
alongside the unchanged v4 model. All320 labels retained, full rawMSE primary;
tail and max-label diagnostics supplementary, never alternative objectives.
No external labels or test-label substitutions. No concentration columns.

Before reading confirmation results, the parent additionally fixes one equal
ensemble: 25% v4 plus25% each of the three discovery-selected family winners.
Do not optimize its weights on confirmation or drop a member after observing
its results. The OOF folds and targets must match exactly before averaging.
This ensemble is an additional hypothesis, not an independent dataset.

A separate no-tuning stress test groups standardized InChIKey14 identities,
so replicate compounds cannot cross folds. It checks validation sensitivity;
its result must not be used to select v5 hyperparameters.

No automatic submission. Keep the default <=3M-each confirmation gate and
report scaffold, all four confirmation results, and any regression. The prior
one-candidate upload approval does not authorize another submission.
