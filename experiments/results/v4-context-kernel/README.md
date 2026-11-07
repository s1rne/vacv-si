# Structure/context product kernel (negative result)

This is a new hypothesis, not a claimed literature reproduction: use
`Tanimoto(Morgan binary)^2 * (0.1 + 0.9 * same known strain/culture)` instead
of the existing additive context kernel. The context equality kernel and the
Tanimoto kernel are positive semidefinite, and their elementwise product is
positive semidefinite. Missing context contributes only the structure term.
Context strings use existing strip/casefold normalization; no semantic assay
merging, target encoding, external records, or test labels are used.

Six profiles were fixed before evaluation: alpha 0.01/0.1/1 crossed with raw
SI/1000 and sqrt(SI). Ridge regression has an unpenalized train-only intercept
and consistently centered train/query kernels. Predictions have a zero floor
but no train-maximum cap; all 320 official labels, including 36066.667, remain.
The discovery seed 30 selected `product-sqrt-1`, then frozen on seeds 31/32
and the existing five-fold scaffold split. The raw70 analog control is rerun
on exactly the same folds. Each OOF file covers every row exactly once.

| Split | Frozen product kernel MSE | Paired raw70 control MSE |
| --- | ---: | ---: |
| Discovery 30 | 4,269,770.99 | 4,338,548.91 |
| Confirmation 31 | 4,220,145.89 | 4,207,602.96 |
| Confirmation 32 | 4,229,253.35 | 3,811,529.21 |
| Confirmation mean | 4,224,699.62 | 4,009,566.09 |
| Scaffold | 4,276,619.97 | 4,290,081.64 |

The product kernel loses 5.37% to paired raw70 on confirmation mean. It does
reduce errors away from the single largest target, but predicts that target
only 209.69 and 187.08 versus raw70's 1273.96 and 2324.92. This is why the
ordinary-row improvement does not improve the actual full raw-MSE objective.
The six-profile discovery grid also shows raw regression causing excessive
ordinary-row predictions without recovering the largest held-out label.

Neither the mean-under-4M plus paired-improvement gate nor the hard 3M-each
confirmation gate passed. No candidate submission was generated or uploaded.
Discovery and confirmation splits reuse a small dataset and do not establish
independent statistical confidence; scaffold results are a separate stress
test. Do not promote this model based on its lower non-tail error alone.

Reproduction, from the repository root (output directory must be absent):

```sh
PYTHONPATH=. DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs /Users/mike/Projects/vacv-si-novel/.venv/bin/python experiments/v4_context_kernel.py
PYTHONPATH=. DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs /Users/mike/Projects/vacv-si-novel/.venv/bin/python -m pytest -q tests/test_v4_context_kernel.py
```

The experiment completed successfully. Three tests passed: kernel PSD with
missing contexts, validation-target isolation, and centered constant-target
intercept in both scales. `manifest.json` records data/source/OOF hashes,
configuration, winner and gates. `metrics.json` contains full-tail metrics for
all discovery profiles and frozen confirmations; no diagnostics replace the
full MSE used for selection.
