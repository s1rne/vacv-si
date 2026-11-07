# Luna v2 external SI model combination

Bounded exploratory paired CV; all official training rows and targets are retained.
The external GLM export has no independent unit-verification proof, so this is experimental evidence only.
No Kaggle upload was attempted. External information is used only for SI regression targets and eligibility filtering.

| Split | Baseline MSE | Frozen augmented model MSE | Paired change |
|---|---:|---:|---:|
| 18 | 4,284,302 | 4,283,778 (external_weight_0.10) | -0.01% |
| 19 | 4,280,237 | 4,280,990 (external_weight_0.10) | +0.02% |
| 20 | 4,272,733 | 4,272,612 (external_weight_0.10) | -0.00% |
| scaffold | 4,283,994 | 4,284,962 (external_weight_0.10) | +0.02% |

Eligible unique external rows: 177.
Source SHA256: `cca80a8f0e29c803be67cbde371294a0f603a3749005b659d8af83314f5293a0`.
Maximum official training target retained: 36,066.667.
The split scores are repeated CV on the same small competition training set, not independent external validation.
