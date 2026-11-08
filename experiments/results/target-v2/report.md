# Official target revision and regression comparison — 2026-11-07

## Root cause

The authenticated Kaggle Data page now provides `train_v2.csv`. The official
download has SHA256 recorded in `data-manifest.json`. Comparing allowed fields
shows identical row order and features, but **81 targets changed from 10000 to
0**. Other 239 targets are unchanged; the test file is byte-identical to the
earlier download. No forbidden competition fields were parsed for this audit.

The old formula p(inactive)*10000 + (1-p)*active_mean optimized the wrong
target. Rescoring saved OOF against the new labels gives 22,874,737 MSE for
original .3/.7 weights, 24,529,786 for hybrid, and 25,071,979 for graph/CheMeleon.
Thus improving the old classifier could make the updated task worse. This
explains the reversal mechanistically; exact organizer revision timing and
hidden-test label changes are not directly observable. No announcement was
present in the competition Discussion page at inspection time.

The historical 14,633,838.88963 submission used context/structure ExtraTrees
(commit ebfaf14), not the later .3/.7 control. Its source used y>=9999 and thus
also treated the real 36066.667 target as inactive. We do not preserve this
mistake in the new regression pipeline.

## New local results

Raw SI MSE, all 320 labels retained, five folds per split. Split6 is discovery,
7/8 confirmation; scaffold is a separate stress test. Preprocessing is fitted
inside each fold. Full OOF and fixed configurations are in the sibling result
directories. These are repeated splits of the same small dataset, not fresh
independent data.

| Configuration | Discovery6 | Mean7/8 | Scaffold |
|---|---:|---:|---:|
| Training-fold mean | 4,283,538 | 4,286,769 | 4,299,946 |
| ET context/structure log | 4,232,994 | **4,241,095** | 4,262,617 |
| CheMeleon ET raw | 4,541,849 | 4,399,599 | 4,549,986 |
| CheMeleon ET log | 4,245,596 | 4,283,416 | 4,261,772 |
| CheMeleon ET log + external presence | 4,213,149 | 4,268,475 | 4,260,691 |
| CheMeleon ridge log | 4,253,475 | 4,276,112 | 4,241,210 |
| Structure ET log | 4,234,663 | 4,272,549 | not run |
| Structure ET log + external measurements | **4,206,976** | 4,253,379 | not run |

Eight initial context/structure/kernel/CatBoost candidates, four embedding
heads and two external-training variants were evaluated. Some raw/kernel
candidates were rejected at discovery; their results were preserved.

Collected data helped modestly: presence features improve paired CheMeleon
confirmation MSE by 0.35%; measured external training improves paired structure
confirmation MSE by 0.45%, with one confirmation split worse. Neither beats
the selected context/structure candidate on mean7/8. External training used
316 unique measured ChEMBL/PubChem molecules, sample weight .25, with all 812
standardized competition connectivity keys excluded (zero overlap). No SI
lookup, manual test annotations or concentration proxies were used.

The 36066.667 training target dominates MSE: in the selected discovery run its
single squared error contributes 4,054,810 to the total 4,232,994. It is kept
in training and every primary score. Therefore tiny CV differences need caution.
Log models are also biased toward smaller values compared with the raw
conditional mean; raw regression alternatives were tested explicitly.

## Actual Kaggle result

Submitted `submission-v2.csv`, SHA256
`d32a9d7ab5bbf11b865b773728f894f3039e37247a8a8b83e7d49111369b9c1b`.
Kaggle status **Complete**, public MSE **2,038,068.29444**.
This is **86.07% below the actual previous team best 14,633,838.88963**.
It is not a comparison against the failed 18M submissions. Private ranking and
current overall leaderboard place were not checked; no claim of first place.

## Reproduction and safeguards

```sh
python scripts/import_competition_v2.py '/path/to/current-kaggle.zip' --out data/raw_v2
PYTHONPATH=. python scripts/submit_v2.py --out outputs/v2/submission.csv
PYTHONPATH=. python experiments/restore_baseline.py --data-dir data/raw_v2 --out outputs/v2-cv --splits 6 7 8 scaffold
PYTHONPATH=. python experiments/v2_embeddings.py --data-dir data/raw_v2 --out outputs/v2-embeddings
PYTHONPATH=. python experiments/v2_external.py
python -m pytest -q
```

New loading requires the official train_v2 filename, whitelists permitted
columns, and records source hashes. Historical sentinel CLIs require explicit
`--allow-legacy-target`; old raw datasets remain untouched. Source, target,
template and CSV hashes accompany exports. The new submission has 597 finite
nonnegative rows in template order. Production/OOF parity is exact for the
selected model. Full suite: **208 passed**. No new dependencies or system
configuration changes were needed. On this Mac use the existing environment's
sklearn/.dylibs in process-local DYLD_LIBRARY_PATH as in previous reports.
