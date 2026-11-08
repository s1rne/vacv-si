# Offline record-level external SI snapshot

This experiment rebuilds pairs directly from existing ChEMBL activity and
assay cache records. It does not read the SI labels of the GLM export; those
tables supply document IDs only. No network calls were made. The source
workspace was not modified.

35,488 distinct cached activity IDs yielded 2,754 individually filtered
records and 170 unambiguous, competition-disjoint pairs: **47 exact, 82 upper
bounds, 41 lower bounds**. Every pair retains both activity IDs, assay IDs,
endpoint types, original/standard value and units, relation, paper and known
matching cell line. Raw record provenance is in `source-records.json`.

Rules: toxicity must be explicitly CC50, antiviral activity must be cellular
VACV IC50/EC50. MCC and other proxy endpoints are excluded. Different assays,
values, endpoints or relations on one side make a pair ambiguous; they are
quarantined, never averaged. Exact duplicate records within the same assay
can collapse. Conflicting cache snapshots remain quarantined even if an
earlier snapshot repeats later. Both-side censoring without an identified
ratio bound is excluded. Competition InChIKey14 and exact-SMILES overlap is
zero; invalid external structures are rejected. One invalid competition
structure uses exact-SMILES fallback, recorded in the manifest.

Example regression fixture: activity 17992015 is VACV EC50 >800 nM and
17991946 is CC50 =7900 nM, same paper/cell; the result is **SI <=9.875**.
The separate MCC =4000 nM record does not contribute. No boundary becomes
an exact SI target.

Three independently inspected exact source pairs are reproduced: activities
1690855/1690830 -> 4155.5556; 2305813/2306046 ->2633.3333;
1690856/1690831 ->2100. The largest exact SI is 157026.3158 from
CHEMBL5223146 activities 24978328/24978308. Their standard and original units
agree: IC50 7.6 nM (=0.0076 uM), CC50 1193400 nM (=1193.4 uM), Vero,
four-day colorimetric assays. This is a source-record ratio, not independent
reverification against the underlying paper.

## Limits

Matching paper and cell line does not prove full assay-protocol equivalence.
Known differing exposure times are rejected; missing times remain explicitly
flagged. Only 31 pairs have exposure times reported on both sides. Treat this
as a conservative experimental snapshot, not 170 fully independently audited
biological measurements. Sources may contain errors even when their unit
representations agree. Unknown original measurement details are not invented.

These records do not provide a close analogue of the 36,066.667 training
compound. Maximum Morgan Tanimoto is 0.368 (an upper bound); maximum among
the near exact examples is 0.333. The 157k compound has similarity 0.209.
Thus higher SI values alone do not establish useful tail coverage.

Run: `PYTHONPATH=. python experiments/v3_source_pairs.py --source /path/to/source-checkout`
with a fresh output directory. Four narrow regression tests pass in
`tests/test_v3_source_pairs.py`. `manifest.json` hashes all used cache files,
the source document-list tables, script and eligible CSV.
