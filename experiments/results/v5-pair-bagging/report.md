# Molecule-weighted pair bagging

Frozen winner: bag-raw.

Confirmation mean is 3,696,435.71 versus paired v4 3,681,220.67: bagging is
0.413% worse. Seed 44 reaches 2.962M, but seeds 42/43/45 fail 3M and the
method does not improve every confirmation. Scaffold improves slightly from
4.254M to 4.248M. Do not promote this variant based on the single good split.
Two tests passed: deterministic symmetric normalized molecule weights and
held-out-target independence including unchanged anchor support. No upload.

| Split | Model | Full MSE |
|---|---|---:|
| 41 | bag-raw | 4017979.394 |
| 41 | bag-log | 4069414.007 |
| 41 | bag-raw-half-v4 | 4113271.524 |
| 41 | v4-ridge-log | 4210499.382 |
| 42 | bag-raw | 3475072.660 |
| 42 | v4-ridge-log | 3347464.347 |
| 43 | bag-raw | 4082005.709 |
| 43 | v4-ridge-log | 4098286.076 |
| 44 | bag-raw | 2962105.083 |
| 44 | v4-ridge-log | 3012754.903 |
| 45 | bag-raw | 4266559.396 |
| 45 | v4-ridge-log | 4266377.334 |
| scaffold | bag-raw | 4248045.239 |
| scaffold | v4-ridge-log | 4253853.162 |

{
  "winner": "bag-raw",
  "profiles": [
    "bag-raw",
    "bag-log",
    "bag-raw-half-v4"
  ],
  "discovery": 41,
  "confirmation": [
    42,
    43,
    44,
    45
  ],
  "draws": 16,
  "rng_seed": 5100,
  "alpha": 10.0,
  "weighting": "sqrt(w_i*w_j), normalized mean one per draw; exponential molecule weights",
  "all_train_molecules_eligible_as_anchors": true,
  "external_data": false,
  "all_labels_retained": true,
  "rows": 320,
  "target_max": 36066.667,
  "confirmation_mean_mse": 3696435.712147125,
  "control_mean_mse": 3681220.665029743,
  "improves_each_confirmation": false,
  "upload_3m_each_gate": false,
  "submitted": false
}

Sixteen fixed exponential-weight draws per training molecule; pair weights are geometric means, normalized to mean one. Pairs, anchors, Ridge alpha 10, log delta target, antisymmetry and fallback exactly follow v4. All outer-training molecules remain anchor-eligible in every draw. Raw mean, log1p mean, and raw mean blended 50/50 with v4 were predeclared; discovery alone selected the frozen profile. Other profiles are not scored on confirmation. Every fold retains all labels including the maximum. No upper prediction caps or external data. Support audit reports anchors and between-draw spread. This repeated CV is not independent public-score evidence. No submission created or uploaded.

Run from repository root: `PYTHONPATH=. python experiments/v5_pair_bagging.py` (output directory must be absent).
Tests: `PYTHONPATH=. python -m pytest -q tests/test_v5_pair_bagging.py`.
