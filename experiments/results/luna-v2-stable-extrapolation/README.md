# Stable v2 extrapolation experiment

This experiment tested four predeclared models with fold-local feature
stabilization: signed `log1p` on RDKit descriptors, clipping those transformed
features to the training fold's 1st/99th percentiles, then fold-local median
imputation, scaling, context encoding, and PCA over frozen CheMeleon embeddings.
Only `SMILES`, `Title`, `strain`, and `Culture` are features. Competition
concentration columns, their missingness, derived values, and external/test
labels are not used. SI targets are not clipped or removed; every score uses all
320 rows, including the 36,066.667 maximum.

Seed 18 selected `kernel-sqrt` (polynomial Kernel Ridge, degree 2, alpha 1,
gamma 1/feature count, coef0 1). Its full raw-MSE was 4,259,505.24. The frozen
winner scored 4,251,891.00 on seed 19 and 4,246,799.77 on seed 20; scaffold
MSE was 4,249,248.87. Both confirmation scores exceed the 3,000,000 gate, so
the candidate is explicitly marked `GATE_FAIL_DO_NOT_SUBMIT` in `manifest.json`.
The failure is dominated by the high-target tail (about 192.1M tail MSE on each
confirmation split); stabilization prevents descriptor-driven numerical
explosions but does not extrapolate the maximum SI well enough.

The complete OOF files contain all 320 target rows for each split. The
597-row candidate CSV is retained only as a finite reproducibility artifact;
it was not submitted to Kaggle. `metrics.json` and `manifest.json` include the
per-split scores and input/script/embedding hashes.

Run from the repository root:

```sh
PYTHONPATH=. DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs /Users/mike/Projects/vacv-si-novel/.venv/bin/python experiments/luna_v2_stable_extrapolation.py
```
