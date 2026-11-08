# V6: alternative pair learning hypotheses

The comparison baseline remains the submitted v4 Ridge-log, public score
1,710,149.26680. No leaderboard inference about competitors' methods is made.
Their scores do not establish what training data or algorithms they used.

## Primary literature and hypotheses

[Potency changes in matched pairs](https://pubmed.ncbi.nlm.nih.gov/25191787/)
compares kernel approaches to predicting changes, motivating a nonlinear
pair-difference kernel. This is not a reproduction of the published pipeline.

[PADRE](https://wenhao-gao.github.io/assets/publications/2021_PADRE_JCIM/paper.pdf)
learns property differences from paired representations. Our background-aware
tree variant retains both fingerprint difference and pair mean, so an identical
chemical change can receive a different correction in different backgrounds.
An antisymmetric prediction wrapper reverses only the difference, preserving
the shared background. Published successes do not establish SI-MSE improvement.

## Protocol

First batch selects each family on discovery51, then freezes and compares
on identical confirmations52–55 plus scaffold:

- Tanimoto pair kernel, powers1/2 and ridge alpha1/10.
- Query-local pair weights, powers4/8 or fixed half-local4 plus half-v4.
- Audited external exact-label training pairs, low fixed auxiliary weights;
  query anchors remain competition training molecules only.
- Stateless character hashing of allowed Title text, appended to pair delta
  features at scales1/3; no source-label reconstruction or text-trained lookup.

After inspecting the first batch, two additional families use discovery56,
then confirmations57–60 and scaffold, each paired with the same v4 baseline:

- Background-aware ExtraTrees pair regression, leaves2/5,400 trees.
- Pair targets log(y+c), fixed c0.01/0.1/10, retaining the original fallback.

No post-confirmation parameter adjustments. All320 labels and original target
values retained. Zero-valued labels remain in training. RawMSE is primary;
diagnostics excluding the maximum are never used as a replacement metric.
All trainable pair endpoints and label statistics come from outer training
folds. Repeated splits are the same dataset, not independent new data.
No Kaggle uploads are part of these experiments.

## External-data boundary

Only the immutable v3-source-pairs audited snapshot is eligible for this batch:
SHA256 `9ead9fc63139b88352584a39673f3458beaf1abb8c3ddd776c0de7d1729a2e23`.
Recheck standardized identity and exact-SMILES exclusion against ALL competition
train/test molecules. Use exact rows only; bounds never become exact targets.
Conservative strain/cell aliases may align descriptions, but missing or
ambiguous conditions stay unknown. Full assay protocol equivalence is still
not established by matching strain/cell alone. GLM's in-progress exports are
not incorporated without their own record-level audit.

## Measured first-batch outcomes

| Family | Mean MSE52–55 | Same-fold v4 | Wins |
|---|---:|---:|---:|
| Pair kernel Tanimoto1/alpha10 | 4,117,073 | 3,595,103 | 0/4 |
| Half local4 + half v4 | 3,755,133 | 3,595,103 | 0/4 |
| External/cross-pair weight0.5 | 3,690,391 | 3,595,103 | 0/4 |
| Hashed Title morphology scale1 | 4,015,883 | 3,595,103 | 1/4 |

No family improves the confirmation mean. Local weighting reduces ordinary-row
MSE but worsens high-SI prediction. External data yields47 exact molecules,
21 with known normalized context,19 participating in training pairs;64 directed
external pairs and0–2 cross pairs per fold. Only17/47 records have verified
matching exposure; full protocol equivalence remains unverified for all47.
No test anchors or bound-to-exact conversions are used.

Offset learning selects c10, but confirmations57–60 average3,936,197 versus
3,410,539 for v4, losing all four and scaffold. Smaller offsets already fail
on discovery: c0.01 gives133.099M and a spurious prediction204,131; c0.1 gives
7.987M. No additional offsets were tried after observing these results.

Background-aware pair ExtraTrees selects leaf5. Confirmations57–60 average
3,920,989 versus3,410,539 for v4 (+14.97%), losing all four. Scaffold is
4,254,462 versus4,253,853. Both ordinary-row and tail errors worsen; retaining
the pair background in this custom tree implementation does not improve v4.

These are repeated splits of the same320 labels. Differences between the two
batches' baseline means reflect different folds, not a change to the baseline.
Retain each negative result and avoid selecting favorable seeds for reporting.

## Selection limitation and next protocol

Single discovery splits repeatedly select candidates that lose confirmation.
The high-SI tail and repeated reuse of320 labels make apparent improvements
fragile. Before another broad search, freeze a small candidate set and use
nested CV: tune within each outer training fold only, then compare outer OOF
against the fixed submitted v4 on the same folds. Never select settings from
outer errors or public feedback. Report scaffold separately and keep repeated
CV distinct from independent test evidence. This protocol is a next-step
proposal, not an executed experiment in this checkpoint.

Finding compatible, disjoint close chemical series remains a data acquisition
hypothesis. Neither sparse external coverage nor competitors' public scores
establish an achievable rawMSE for a structure-only learner on these labels.
