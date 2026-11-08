# Nonlinear RKHS pair differences

Frozen winner: pair-kernel-p1-a10.

Negative result: confirmation mean 4,117,072.56 versus paired v4 3,595,103.06,
14.52% worse. Every confirmation and scaffold loses to v4. The nonlinear
kernel reduces the learned amplification of the largest target's anchors:
seed 54 predicts 1190.79 versus v4 7601.00 (true target 36066.667), explaining
much of its MSE deficit. The 3M-each upload gate fails. No promotion or upload.

Three tests passed: pair-kernel PSD/antisymmetry; numerical equivalence of
unique-pair alpha/2 to directed-pair alpha; held-out-target independence.

| Split | Model | Full MSE |
|---|---|---:|
| 51 | pair-kernel-p1-a1 | 4220063.280 |
| 51 | pair-kernel-p1-a10 | 4128956.671 |
| 51 | pair-kernel-p2-a1 | 4244318.266 |
| 51 | pair-kernel-p2-a10 | 4178211.482 |
| 51 | v4-ridge-log | 4089231.368 |
| 52 | pair-kernel-p1-a10 | 4075781.100 |
| 52 | v4-ridge-log | 3886034.771 |
| 53 | pair-kernel-p1-a10 | 4049081.480 |
| 53 | v4-ridge-log | 3560667.962 |
| 54 | pair-kernel-p1-a10 | 4125682.136 |
| 54 | v4-ridge-log | 2746470.846 |
| 55 | pair-kernel-p1-a10 | 4217745.506 |
| 55 | v4-ridge-log | 4187238.661 |
| scaffold | pair-kernel-p1-a10 | 4276848.618 |
| scaffold | v4-ridge-log | 4253853.162 |

```json
{
  "winner": "pair-kernel-p1-a10",
  "profiles": {
    "pair-kernel-p1-a1": [
      1,
      1
    ],
    "pair-kernel-p1-a10": [
      1,
      10
    ],
    "pair-kernel-p2-a1": [
      2,
      1
    ],
    "pair-kernel-p2-a10": [
      2,
      10
    ]
  },
  "discovery": 51,
  "confirmation": [
    52,
    53,
    54,
    55
  ],
  "regularization": "unique undirected pairs use alpha/2; equivalent to both orientations at alpha",
  "selection": "minimum full raw MSE on discovery only",
  "external_data": false,
  "all_labels_retained": true,
  "rows": 320,
  "target_max": 36066.667,
  "confirmation_mean_mse": 4117072.5553157507,
  "control_mean_mse": 3595103.0599298924,
  "improves_each_confirmation": false,
  "upload_3m_each_gate": false,
  "submitted": false
}
```

Molecular kernel K=Tanimoto^power, power 1/2; pair kernel K(a,c)-K(a,d)-K(b,c)+K(b,d). Four profiles cross powers 1/2 with ridge alpha 1/10, chosen on seed 51 then frozen. Unique undirected pairs use alpha/2, equivalent to the doubled residual objective of both orientations at alpha. Antisymmetry follows directly from the pair kernel; no intercept. Pair selection, log1p targets, top3 anchors at similarity>=0.7, raw-space anchor aggregation, zero floor and structural log-KNN fallback exactly follow v4. All training labels retained; no upper prediction caps, external data or forbidden features. Held-out labels never enter pair targets, anchor values or any fit. Repeated CV is not independent leaderboard evidence. No submission created or uploaded.

Run from repository root: `PYTHONPATH=. python experiments/v6_pair_kernel.py` (output directory must be absent).
Tests: `PYTHONPATH=. python -m pytest -q tests/test_v6_pair_kernel.py`.
