# Recovery checkpoint — 2026-11-07

Base: main 66b429f; branch codex/recovery-clean-cv. Actual reruns, raw SI MSE,
320 training rows, fixed five-fold KFold splits, seven forest seeds.

## Same-input comparison

All candidates use allowed structure/context plus target-independent external
record counts and molecular similarities. No recovered SI, document prior,
manual value, forbidden competition concentration, or test target is used.

| Model | Split 3 | Mean confirmation 6–8 |
|---|---:|---:|
| Original architecture/weights (context .3, grouped .7) | 9,503,958 | 9,951,383 |
| Clean hybrid | 8,545,619 | 8,953,685 |
| Graph75 + CheMeleon25 | 8,144,236 | 8,836,244 |
| Hybrid, probability exponent 1.2 | 8,318,421 | 8,821,410 |

The calibrated hybrid is selected: 11.35% lower confirmation MSE than the
original-weight control, 1.48% lower than the clean hybrid, and only 0.17%
lower than graph/CheMeleon. This small difference is not statistical proof of
superiority. Splits share the same small dataset; they are not independent
new data. Model selection used the confirmation comparison too.
The calibrated model's scaffold MSE is 12,362,233 versus 12,348,388 for the
uncalibrated hybrid (0.11% worse). Active-value kNN did not help and was rejected.

This is an architecture comparison under the same repaired feature policy,
not a byte-for-byte rerun of the historical submission. Earlier 9,766,480 /
8,634,637 figures used different lookup artifacts and cannot establish an
honest improvement over the current public leaderboard.

## Fixed integrity issues

- CSV loaders whitelist allowed columns before parsing; test targets never load.
- Clean feature assembly preserves input order and drops all targets.
- Recovered lookup CV rebuilds publication choices and all supervised rules
  inside each outer training fold; absence of the external corpus is an error.
- Substitution advice no longer uses the 6M reference derived from test labels.
- CV and submission CLIs default to clean inputs; legacy active-value checkpoint
  mode fails explicitly because it requires excluded concentration features.

The original external corpus is absent locally. Real recovered-lookup rebuilding
has therefore not been exercised; its wiring is covered by synthetic invariance
tests. Production clean predictions do not depend on that corpus or path.

## Artifacts and verification

Selected file: `../recovery-candidates/submission.csv`, SHA256
`f6388a21766965ba6d2611c6b1fa767a439101b3d41de1b6a6efea6f57033b5b`.
Graph alternative: `submission.csv`, SHA256
`ea0edab20d184b9694862e002f9f36eb2a0672639402d88470dd6ae7e6a4e7cf`.
Both have 597 finite nonnegative predictions in exact template order.
Full suite: 199 tests passed; additional targeted checks run after metadata and
index-preservation changes. Source and input hashes are stored beside CSVs.

```sh
PYTHONPATH=. python experiments/recovery_checkpoint.py --splits 3 6 7 8 --out outputs/recovery-clean
PYTHONPATH=. python scripts/submit_checkpoint.py --lookup-policy clean --out outputs/graph/submission.csv
PYTHONPATH=. python scripts/submit_recovery.py --out outputs/calibrated/submission.csv
python -m pytest -q
```

On macOS the existing local environment requires DYLD_LIBRARY_PATH pointing to
its sklearn/.dylibs directory. No system library installation was performed.

## Kaggle investigation

The authenticated submission history identifies 18,459,829.01365 as the separate
`cfff85d` ExtraTrees/CheMeleon/external auxiliary model, not the graph checkpoint.
The visible historical best is 14,633,838.88963. No evidence yet establishes
whether older submissions were rescored. Local files match the supplied ZIP;
whether Kaggle's files changed remains unverified. Public result for this upload
is recorded separately in `kaggle-result.json` once scoring completes.

Actual Kaggle upload completed: **18,146,452.74419** public MSE, 1.70% below
the recent 18,459,829.01365 submission but worse than the historical visible
best 14,633,838.88963. This candidate is NOT the team leaderboard best. Local
CV improvements did not transfer; further validation/domain-shift investigation
is required before claiming improvement over the original leaderboard model.
