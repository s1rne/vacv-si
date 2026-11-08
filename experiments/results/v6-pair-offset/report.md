# Log-target offset in molecular pair regression

Frozen winner: offset-10.

Negative result: confirmation mean 3,936,196.61 versus paired v4 3,410,538.72,
15.41% worse. All four confirmations and scaffold lose to v4. The smaller
offsets caused extreme predictions already on discovery: c=0.01 scored
133.099M with maximum prediction 204130.58; c=0.1 scored 7.987M. Increasing
offset to 10 controls this amplification but weakens useful tail prediction.
For seed 59, largest-target prediction is 2504.11 versus v4 8473.39. No
adaptive extra offsets were tried. The upload gate fails; no promotion.

Three tests passed: c=1 parity with v4 including structural fallback;
held-out target independence; inverse recovery of zero and small values.

| Split | Model | Full MSE |
|---|---|---:|
| 56 | offset-0.01 | 133098844.831 |
| 56 | offset-0.1 | 7987475.512 |
| 56 | offset-10 | 4075142.466 |
| 56 | v4-ridge-log | 4048951.271 |
| 57 | offset-10 | 3954799.164 |
| 57 | v4-ridge-log | 3670751.986 |
| 58 | offset-10 | 4141901.239 |
| 58 | v4-ridge-log | 4085042.386 |
| 59 | offset-10 | 3718043.441 |
| 59 | v4-ridge-log | 2573888.892 |
| 60 | offset-10 | 3930042.586 |
| 60 | v4-ridge-log | 3312471.624 |
| scaffold | offset-10 | 4264285.236 |
| scaffold | v4-ridge-log | 4253853.162 |

```json
{
  "winner": "offset-10",
  "profiles": {
    "offset-0.01": 0.01,
    "offset-0.1": 0.1,
    "offset-10": 10.0
  },
  "discovery": 56,
  "confirmation": [
    57,
    58,
    59,
    60
  ],
  "transform": "log(y+c), inverse exp(z)-c; structural fallback unchanged log1p",
  "selection": "minimum full raw MSE on discovery only",
  "external_data": false,
  "all_labels_retained": true,
  "rows": 320,
  "target_max": 36066.667,
  "confirmation_mean_mse": 3936196.6074323226,
  "control_mean_mse": 3410538.722002673,
  "improves_each_confirmation": false,
  "upload_3m_each_gate": false,
  "submitted": false
}
```

Fixed offsets c=0.01,0.1,10 modify only pair target and anchor transforms. All other v4 behavior remains: Ridge alpha10, both pair orientations, antisymmetric difference predictions, same-context top3 anchors >=0.7, raw-space aggregation and unchanged structural log1p fallback. Floor at zero only after anchor aggregation; no upper cap. All 320 labels including zeros and maximum retained. No external or forbidden features. Discovery chooses once; four confirmation splits and scaffold use the frozen winner and unchanged v4 control. No submission generated or uploaded. Repeated CV is not independent leaderboard evidence.

Run: `PYTHONPATH=. python experiments/v6_pair_offset.py` (output must be absent).
Tests: `PYTHONPATH=. python -m pytest -q tests/test_v6_pair_offset.py`.
