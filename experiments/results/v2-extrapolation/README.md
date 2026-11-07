# Extrapolating regression: rejected candidate

Four fixed profiles tested whether genuine extrapolation can overcome the
train-target-range lower bound: Ridge(alpha10) and degree2 polynomial
KernelRidge(alpha1), each predicting raw SI or sqrt(SI). Polynomial gamma is
1/feature count and coef0=1; target centering uses the training mean. Sqrt inverse
is max(prediction,0) squared. No upper clipping or per-row handling is applied.

Features combine actual RDKit physicochemical descriptors, context, and16 frozen
CheMeleon PCA components. Median imputation, scaling, context fitting and PCA are
trained independently in every fold. All320 official targets remain present;
there are no external labels or prohibited competition features.

| Profile | Discovery15 full MSE |
|---|---:|
| Ridge raw | 165,203,676 |
| Ridge sqrt | **4,254,412** |
| Polynomial raw | 1.256e12 |
| Polynomial sqrt | 6.740e21 |

The frozen discovery winner Ridge sqrt fails confirmation16/17 catastrophically:
MSE1.673e16 and5.012e12; scaffold4,244,658. The maximum-label predictions remain
only197/253, while unrelated low-target molecules receive enormous predictions.
This fails the requirement of<=3M on both independent confirmation seeds.
No production baseline or Kaggle submission was run. The saved candidateCSV is
strictly a reproducibility artifact and must not be submitted.

`instability-diagnostics.json` identifies a feature-space extrapolation problem:
row232 (target3.2857) has RDKit Ipc2.876e16, versus a fold-training maximum4.396e12.
After fold-local scaling, this becomes99,612 standard deviations on seed16 and
86,786 on seed17. Squaring extrapolated sqrt predictions amplifies the error;
predictions reach2.314e9 and4.005e7. This is distinct from successful target-tail
extrapolation. No feature transform, threshold, or model was tuned on these
confirmation failures.

Artifacts include full OOFs, metrics, source/data/embedding hashes, and a full597-row
candidateCSV. Validation confirms all320 targets including36k, complete five-fold
coverage, finite nonnegative outputs, and exact sample-submission schema/order.

Run from repository root with an empty output directory:

```sh
DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs PYTHONPATH=. /Users/mike/Projects/vacv-si-novel/.venv/bin/python experiments/v2_extrapolation.py
```
