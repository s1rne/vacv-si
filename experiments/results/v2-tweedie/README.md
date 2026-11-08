# Official v2 Tweedie experiment

Two frozen CatBoost configurations (variance power 1.5/1.8), with fold-local
context and structural features. No external data or forbidden fields. All
320 targets retained, including zero and 36,066.667; division by 1,000 is a
fixed unit conversion, reversed at inference.

Seed9 selected power1.8 (4,132,970 MSE). Frozen confirmation: seed10
4,194,234; seed11 4,277,813; scaffold 4,271,090. These are full raw MSE.
Headline results are not tail-excluded diagnostic scores.

Kaggle accepted the 597-row file and returned **2,062,945.32569**. This is
worse than the previous best 2,038,068.29444; do not replace that best.
The model is a completed negative experiment, not a demonstrated improvement.

`verification.json` records exact template ordering and zero change after
injecting forbidden fields and target fields into training/query inputs.
`manifest.json` records input, source and submitted-file hashes.

Run from the repository root with an empty output directory:
`PYTHONPATH=. python experiments/v2_tweedie.py`.
