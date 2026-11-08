# Official v2 target regression — 2026-11-07

The original 14.634M public model was reconstructed from commit `ebfaf14`:
seven seeds of 800-tree ExtraTrees classifiers, leaf size 2, 80% context and
20% molecular structure. It used `y >= 9999` as an inactive label; newer models
used exact `y == 10000`. Feature definitions are byte-identical to that commit.
Old-data reproduction was stopped when the parent downloaded official
`train_v2.csv`: the task changed from sentinel classification to actual SI
regression. Never use the old competition concentration columns to derive labels.

All models below read only permitted SMILES, Title, strain, Culture and the
official training SI target. There are no external presence features or SI
lookups. Train targets, including the largest 36066.667 value, are kept intact.

## Fixed screening and confirmation

Discovery: fixed random fivefold split seed 6. Eight predetermined profiles:

| Model | Raw-SI MSE |
|---|---:|
| Constant training mean | 4,283,538 |
| ExtraTrees context, raw target | 4,248,044 |
| ExtraTrees context/structure 50/50, raw target | 4,289,875 |
| Kernel ridge, raw target | 4,541,786 |
| ExtraTrees context, log1p target | 4,241,281 |
| **ExtraTrees context/structure 50/50, log1p target** | **4,232,994** |
| Kernel ridge, log1p target | 4,266,519 |
| CatBoost, raw target | 4,323,129 |

The winning architecture was frozen before seeds 7 and 8. Three forest seeds,
800 trees per forest, leaf size 2, max_features 0.5. Average predictions in log
space across context and structure forests, then expm1 and nonnegative floor.
The production model matches discovery fold 0 predictions exactly (max_abs 0).

| Confirmation | Selected model | Constant training mean |
|---|---:|---:|
| Random 7 | 4,242,483 | 4,281,978 |
| Random 8 | 4,239,708 | 4,291,560 |
| Mean random 7–8 | 4,241,095 | 4,286,769 |
| Scaffold | 4,262,617 | 4,299,946 |

Confirmation improvement over constant: 1.065%; scaffold: 0.868%. These are not
comparisons to public 14.634M; only a real Kaggle upload can establish that.
The maximum training target accounts for 94.53% of sum(y²), and contributes
4.055M to discovery MSE. Excluding it solely for diagnosis gives 178,743 MSE
on the other rows. It was never excluded from fitting, selection, or headline MSE.

## Reproduce

Run from the repository with its existing environment. macOS needs the existing
sklearn/.dylibs process DYLD_LIBRARY_PATH setting, as in the root README.

```sh
PYTHONPATH=. python experiments/restore_baseline.py --data-dir data/raw_v2 --out outputs/v2-screen --splits 6
PYTHONPATH=. python experiments/restore_baseline.py --data-dir data/raw_v2 --out outputs/v2-confirm --profiles et-blend-log constant --splits 7 8 scaffold --submit
PYTHONPATH=. python scripts/submit_v2.py --out outputs/submission-v2.csv
python -m pytest -q tests/test_regression_v2.py
```

Six focused tests passed: allowed-field isolation, finite/nonnegative target
validation, retaining the full tail and measured 10000, and invariance to changed
query targets/forbidden fields. CSV: 597 finite nonnegative predictions in exact
sample template order. Final file `submission-v2-final.csv`, SHA256
`d32a9d7ab5bbf11b865b773728f894f3039e37247a8a8b83e7d49111369b9c1b`.
The earlier `submission-v2.csv` has identical predictions; final metadata was
regenerated after removing an unnecessary numeric 10000 input guard. Version
checking uses the official `train_v2.csv` filename, not individual numeric values.

`v2/` contains discovery OOF/metrics; `v2-confirm/` contains confirmation and
scaffold OOF/metrics. The discovery run was interrupted after seed 6 to freeze
the winner and avoid screening the unused variants on confirmation seeds.
The confirmation process completed every fold and exported predictions, then
its old in-memory metadata code looked for `train.csv` instead of `train_v2.csv`.
The source path was corrected and metadata regenerated separately; the exported
chosen CSV hash exactly matches the separately successful production CLI run.
