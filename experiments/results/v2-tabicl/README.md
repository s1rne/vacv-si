# TabICL v2 checkpoint

Two fixed configurations use raw SI or log1p SI. The input combines permitted
ContextEncoder features with PCA32 of frozen CheMeleon embeddings. Both context
encoding and PCA fit only on each training fold. All 320 official-v2 targets,
including SI 36066.667, remain in the scoring metric.

Model: `TabICLRegressor`, public standard checkpoint
`jingang/TabICL:tabicl-regressor-v2-20260212.ckpt`, four estimators, CPU,
four Torch threads, no AMP. No competition IC50/CC50 fields, external target
lookup, manual labels, presence features or test-label access.

| Configuration | Discovery seed9 MSE | Confirmation10 | Confirmation11 | Scaffold |
|---|---:|---:|---:|---:|
| raw | 4,321,278 | not selected | not selected | not selected |
| log | 4,273,040 | 4,278,480 | 4,285,411 | 4,283,062 |

The discovery winner, log, was fixed before confirmation. Confirmation mean is
4,281,945; this experiment does not support replacing the current ET candidate.
Raw predictions produce false positive peaks up to 8689 on discovery folds.
No Kaggle submission was sent for this model. A complete candidate CSV is saved
for reproducibility, with all 597 test IDs in sample-submit order.

The full run, including checkpoint download, took about 74 seconds. Validation
passed: complete OOF coverage across four splits, five-fold coverage, retention of
all targets, finite nonnegative outputs, and exact submission schema and ID order.
The pretrained checkpoint remains in the standard Hugging Face cache, outside git.

Run from repository root with an empty output directory:

```sh
DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs PYTHONPATH=. /Users/mike/Projects/vacv-si-novel/.venv/bin/python experiments/v2_tabicl.py
```
