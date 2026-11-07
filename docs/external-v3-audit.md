# External SI audit: source records before model fitting

The `external_v2` export at source commit `31624e0` is not approved wholesale
for the next training experiment. Successful unit conversion does not prove
that different biological endpoints or assays can be pooled.

## Verified defects

`_recomputed_halves` in the external producer collects records by molecule and
cell or publication, averages their values in logarithmic space and declares
the relation exact when any contributing record is exact. Consequently:

- A CC50 value of 7.9 and a distinct MCC value of 4 become about 5.621.
- An exact value of 84.1 and a censored limit of at least 100 become about
  91.706, incorrectly labelled as an exact value.

These changes represent endpoint selection/aggregation changes, not demonstrated
unit-conversion corrections. The current cleaned export contains 34 rows
marked `unit_audit_corrected`. Even rows without this marker need compatible
source records; the word `strict` currently means matching publication and
cell, not necessarily matching exposure conditions or endpoint semantics.

## Source-record check

For publication CHEMBL4000146 and one HEL-cell compound, cached records contain:

| Activity | Assay | Endpoint | Reported value |
|---|---|---|---|
| 17992015 | CHEMBL4000329 | VACV EC50 | >800 nM |
| 17991946 | CHEMBL4000324 | CC50 | 7900 nM |
| 17991932 | separate cytotoxicity record | MCC | 4000 nM |

The explicitly matched CC50/EC50 pair provides an upper SI bound of 9.875.
It does not provide an exact SI. MCC must not replace or average into CC50.

## New experiment contract

- Build from cached source activities, preserving activity and assay IDs,
  publication, endpoint, raw units/relation and converted concentration.
- Use explicit CC50 and cellular VACV IC50/EC50, a shared molecule,
  publication and known cell line. Reject ambiguous repeated assays or
  incompatible known durations; record missing duration as a limitation.
- Exclude all competition molecules, invalid structures and uncertain matches.
- Keep exact measurements and one-sided SI bounds distinct. Do not convert
  doubly censored pairs into an exact ratio.
- Fit exact-only versus exact-plus-interval observations on identical folds.
  Interval loss is zero inside the feasible interval; it must not pull every
  observation to its boundary.
- Keep every official competition target, including 36,066.667. The upload
  gate remains full raw-MSE <=3M on each of two frozen confirmation splits.

This is a conservative database-record audit, not independent replication of
the underlying biological experiments. No test-label substitution is allowed.
