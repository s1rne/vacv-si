# Evidence needed for the next v2 experiment

No Kaggle upload until full raw-MSE is at most 3M on each of two frozen
confirmation splits, with all 320 official labels retained. Public best is
2,038,068.29444; the latest Tweedie candidate did not improve it.

## Why more trees cannot meet this gate

The maximum official training label is 36,066.667; the next largest is 5,000.
When the maximum is held out, any predictor bounded by the remaining training
label range contributes at least (36,066.667 - 5,000)^2 / 320 = 3,016,055.62
to full OOF MSE. This already exceeds the gate even if all other predictions
are perfect. The context analog model has the same limitation and activity
cliffs additionally damage ordinary rows. It is not a sufficient final model.

## Audit feedback for the external-data producer

The first external_v2 export contains 307 measured molecules. Its unit audit
verified zero original records. At least one row with SI approximately 153,050
and notes `si_outside_window` and `pair_spans_documents` still appears in
measured.csv. Routing a flagged row to a temporary censored collection is not
enough if its `measurement_type` remains exact and export partitions on that
field alone. Separate measurement relation from eligibility for training;
quarantine quality failures, rather than turning them into artificial bounds.

There are 60 legacy tier-B rows without independently verified pairs. The
conservative A/E, strict/document, empty-notes subset contains 210 molecules,
including seven with SI >= 1,000. These counts describe a snapshot, not proof
that the units and experimental conditions have been reconciled. Recheck
canonical and exact-SMILES exclusion against every competition molecule;
one competition structure fails standardization.

Prioritize restoring the source-record join for the unit audit and verifying
the high-SI rows first. Preserve original values, units, relation operators,
virus, cell line, document/assay identifiers and DOI. Do not combine endpoints
across viruses, cell lines or incompatible assays. Censored and qualitative
observations must not become exact numeric regression targets.

## Targeted external expansion

The extreme training compound is titled `8e-as-555-15-16`, with permitted
training SMILES:

`CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](NC(=O)C1=CC=C([N+](=O)[O-])C=C1)C2`

Search independent measurements of other compounds in the same chemical
series, not answers for competition molecules. Candidate source identified
by literature research: Sokolova et al., ChemMedChem 2022,
DOI 10.1002/cmdc.202100771. Verify its experimental table and supporting
structures before importing: virus columns, concentration units and exact
versus censored selectivity indices are essential. An entry only helps this
experiment if its standardized molecule is absent from both train and test.

Follow-up source review found that all 47 locally extracted table rows (40
unique standardized molecule keys) already overlap competition molecules.
Thus this known table currently contributes **zero disjoint training rows**.
Its large SI examples must not be added to bypass the exclusion or substitute
test answers. The source table uses micromolar concentrations and separate
virus columns. Reported SI can differ from the ratio of rounded endpoint
means; do not silently replace the official training target with that ratio.

The current 210-row conservative external subset has no close structural
neighbor for the extreme training compound (maximum Morgan Tanimoto 0.333).
Its high target values alone therefore do not establish useful evidence for
this series. Seek chemical coverage, not merely more rows or larger targets.
