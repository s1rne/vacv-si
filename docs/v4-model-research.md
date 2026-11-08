# Activity-cliff experiments and validation checkpoint

## Evidence behind the experiments

[Exploring QSAR models for activity-cliff prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC10107580/)
compares ECFP, descriptor and graph representations with RF, nearest-neighbor
and neural models. Knowing one member's activity improves cliff detection.
This motivates a train-only anchor plus learned activity difference; our
Ridge-log experiment is a custom hypothesis, not a reproduction of the paper.

[mmpdb](https://github.com/rdkit/mmpdb) identifies matched molecular pairs and
estimates property changes from local chemical replacements. Our RDKit
single-cut expert requires repeated transformations in matching assay context.
It is narrower than mmpdb and uses no external labels. Published activity
results do not guarantee performance on raw selectivity-index MSE.

## Frozen experiments

Each experiment chooses its profile on one discovery split, freezes it, then
evaluates two confirmation splits and scaffold CV. All 320 official v2 targets,
including 36,066.667, remain unchanged. Only allowed structure/context inputs
are used. Pair endpoints, anchor labels and transformation statistics belong
to the outer training fold. No test-label reconstruction or Kaggle uploads.

| Method | Confirmation seeds | Mean full MSE | Same-fold context-raw70 |
|---|---|---:|---:|
| Product context/structure kernel | 31/32 | 4,224,700 | 4,009,566 |
| Pair-delta Ridge-log | 34/35 | 3,621,938 | 3,920,150 |
| Single-cut MMP transfer | 37/38 | 4,153,155 | 4,153,158 |

These different seed pairs are not a common leaderboard. Compare each model
only with its paired control. Repeated splits reuse the same 320 observations;
they are not independent new data or evidence of a Kaggle score.

Pair-delta improves seed34 from 3,778,835 to 3,132,248 but worsens seed35 from
4,061,465 to 4,111,629. Its held-out maximum-target prediction is 5,919 versus
786 respectively: useful extrapolation is strongly dependent on fold contents.
Scaffold MSE improves from 4,290,082 to 4,253,853. The initial mean crosses 4M,
but improvement on both confirmations is absent. It fails the unchanged
upload gate of <=3M on each confirmation split.

MMP covers only 13–16 confirmation rows, all with target <=58.824; it cannot
address the high-activity tail. Product kernels improve ordinary predictions
but underpredict the maximum target. These negative findings are retained.

## Reproduction and next evidence

From the repository root with its scientific Python environment:

```sh
PYTHONPATH=. python experiments/v4_context_kernel.py
PYTHONPATH=. python experiments/v4_pair_delta.py
PYTHONPATH=. python experiments/v4_mmp_transfer.py
python -m pytest -q
```

Scripts refuse to overwrite existing result directories. Use an isolated
checkout without generated output directories to replay. Recorded manifests
include source/data hashes; result directories contain full OOF predictions.

The frozen pair-delta model receives additional validation on predeclared
seeds 13/14 and 39/40, with no parameter selection from those results.
Measured pair-delta/control MSE: seed13 4,094,048/4,167,764; seed14
2,178,511/3,886,398; seed39 4,023,416/4,117,446; seed40 3,524,387/4,010,315.
The mean is 3,455,090 versus 4,045,481, a 14.59% paired improvement. All four
improve, but two still exceed 4M. Seed14's held-out maximum prediction rises
to 11,145 while seed13 predicts only 743. This does not erase the failure on
seed35 or establish the upload gate. No parameters were changed after these
results were observed.
Across all six non-discovery random splits (13/14/34/35/39/40), mean MSE is
3,510,706 versus paired control 4,003,704, a 12.31% reduction. Five of six
improve. This combined result includes the failed seed35, not only successes.

Checkpoint verification: `python -m pytest -q` passed 243 tests. The parent
independently recomputed all 37 reported v4 model/split metrics from saved OOF,
checked 320 ordered labels and all five folds, finite nonnegative predictions,
and matched every recorded source hash. Read-only precommit inspection found
no blocking findings.
See `experiments/results/v4-pair-delta-validation/` for that separate check.
The GLM task is in `docs/glm-near-analogs-v4.md`: repair measurement pairing
and seek genuinely new close analogues, including the complete weak/strong
series rather than only high-SI records.
