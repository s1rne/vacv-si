# Single-cut MMP transfer

## Measured results

| Split | Model | Raw MSE | Paired context control | Covered rows |
|---|---|---:|---:|---:|
| 36 | mmp-log | 3783263.842 | 3783267.407 | 16/320 |
| 36 | mmp-raw | 3783263.839 | 3783267.407 | 16/320 |
| 37 | mmp-raw | 4298254.273 | 4298257.898 | 13/320 |
| 38 | mmp-raw | 4008055.193 | 4008057.836 | 16/320 |
| scaffold | mmp-raw | 4290081.636 | 4290081.636 | 0/320 |

Frozen winner: mmp-raw. Research gate: False. Kaggle gate: False.
No candidate CSV or Kaggle upload. All targets retained without clipping or removal.

Confirmation mean MSE is 4,153,154.733 versus paired control 4,153,157.867: improvement only 3.134 MSE units (0.000075%). This is not a meaningful improvement and fails the <4M research gate. Covered targets never exceed 58.824, with zero coverage of targets >=1000 and zero scaffold coverage. The expert cannot repair the dominant high-SI errors in this dataset. All 25 fold/profile audit entries have disjoint transformation-label and validation row sets. Focused pytest: 2 passed.

## Method and limitations

Actual RDKit single non-ring bond cuts, connected core >= half original heavy atoms, variable <=8 heavy atoms; atom-mapped attachment retained. Canonical pairs require identical core and known strain/culture. Directed variable substitutions require two unique unordered training pairs within that same context. Mean transformed-target difference is shrunk n/(n+5), added to a training anchor, then inversed; support-weighted arithmetic mean of proposals. Missing transfer falls back to paired context-raw70. No query targets enter the fitting API. All 320 rows have outer-fold predictions; discovery only selects between two fixed profiles before confirmations. Scaffold is a stress test, not a tuning split.

This is a narrow custom MMP expert, not a reproduction of mmpdb's full multi-cut workflow. Sparse 320-row SI data and assay context restrictions may leave too few repeated transformations. Literature on larger activity datasets does not establish raw-SI score improvements here. Supports can share molecules and are not independent experimental replicates. No external labels or forbidden columns used.

## Primary sources

- mmpdb: https://pubs.acs.org/doi/10.1021/acs.jcim.8b00173 ; official implementation https://github.com/rdkit/mmpdb
- MoleculeACE activity-cliff evaluation: https://pmc.ncbi.nlm.nih.gov/articles/PMC9749029/
- Activity-cliff prediction comparison: https://pmc.ncbi.nlm.nih.gov/articles/PMC10107580/
