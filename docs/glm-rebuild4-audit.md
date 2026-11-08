# GLM rebuild4 audit: main03ca819

Decision: **43 record-supported exact pairs accepted for the requested controlled43-versus47 ablation, with zero new molecular identities**. This rebuild repairs the legacy aggregate-label export bug. It does not establish that43 is a better dataset than our47, nor that missing assay conditions are equivalent.

Accepted snapshot: `experiments/results/glm-rebuild4-audit/accepted.csv`.

SHA256: `dddba1a7dc121b7d5baf6e4b5a4dc3eea8369dd85557e1e5478308fdc0e48a01`.

## Current exports and direct record checks

| Export | Rows | Finding |
|---|---:|---|
| measured_clean |43|All exact, all record_pair_verified;42 SI agree with our47, one alternate measurement|
| censored |126|123 reconstructed numeric bounds, plus3 qualitative not_active rows|
| quarantine |986|Includes the previous exposure conflict and4 valid record pairs excluded by legacy variance flags|

Independent standardization against **all competition train and test SMILES** found zero exact-SMILES or key14 intersections in each export, no stored-key discrepancies. Export hashes match the source manifest. Only structural competition columns were read for this audit.

Read original locally cached ChEMBL activity records for170 pairs:43 exact,123 bounds, and the4 omitted exact pairs;29 documents. Every listed activity ID, assay ID, document ID, endpoint, relation, converted concentration, ratio and within-pair molecular ID matched. There were zero missing records or numerical discrepancies. Phase parsing of these raw descriptions found no known conflicting endpoint-time sets or ambiguous parsed phases. Cached record extracts and hashes are retained in `raw-records.json`; row-level checks are in `raw-checks.json`.

All123 numeric bounds have identities already present in our older source-pair snapshot. They remain bounds, not exact labels. Three qualitative rows were not approved as numeric labels.

Of43 exact pairs,17 have endpoint exposure known on both halves;26 have incomplete exposure evidence. The latter retain the same uncertainty policy as our older snapshot: record-level ratios may be used with an explicit uncertainty flag, but **unknown exposure is not proven assay equivalence**. Known cell identity and a common paper also do not establish full protocol equivalence.

## Producer repair verified

`base_candidates` now reconstructs labels from `record_pair` rather than retaining old corpus SI. Legacy structures/documents nominate source records; the old numerical labels do not become training targets. Reconstruction restricts antiviral endpoints to IC50/EC50 and cytotoxic endpoints to CC50, rejects ambiguous halves, and records activity/assay/document provenance. Failed reconstruction is marked not_record_verified and quarantined. Deduplication prefers verified record pairs over unverifiable legacy candidates. Actual43/123 numeric exports contain the record-verification markers and match cached records.

The former MWOIGQWTBZSCGB issue is repaired: activities18262277/18262178 have48h/24h endpoints despite shared4h preincubation. This identity is now quarantined with different_exposure_times. Focused tests include legacy export rejection and phase-aware exposure: **18 passed**.

Residual general limitation: `_exposure_compatible` accepts intersecting endpoint sets, not necessarily equal singleton sets. None of the170 audited pairs has a conflicting or ambiguous endpoint-set result, so this did not affect the accepted snapshot; future multi-time datasets still need conservative phase/condition handling.

## NIOCH-14 is an alternate measurement, not a correction

The one changed value is YDHJUPUAMAWEOT:

- Our snapshot:157026.3157894737, CHEMBL5223146, IC activity24978328=7.6nM, CC activity24978308=1193400nM;4-day Vero/colorimetric descriptions.
- Rebuild4:149175, CHEMBL6092128, IC activity29121469=8nM, CC activity29121496=1193400nM; both4-day Vero neutral-red dye uptake descriptions, antiviral Copenhagen.

The new value is reproducible within its own document and assay context. It is not the old153050 interdocument aggregate, and no evidence here makes it a universally more correct target than the older157026 measurement. Keep the two provenance claims distinct.

## Why the4 omitted old rows do not prove our snapshot was wrong

| Identity | Old SI | Reconstructed provenance | Rebuild quarantine reason |
|---|---:|---|---|
| DPIOHFGAQQGXIS |970|1690858/1690833, CHEMBL1139008|legacy high_variance; spread1.48|
| FLYFTBZHRIAXCH |0.84|12479151/12479091, CHEMBL3045530|legacy high_variance; spread1.68|
| JEYKGEGRFZIHCQ |104.1666667|1999557/1999579, CHEMBL1138246|legacy high_variance; spread1.50|
| MAOCCGZSMYLPJX |0.1142857|12479153/12479093, CHEMBL3045530|legacy high_variance; spread2.40|

All4 are still `record_pair_verified`, match their cached records, and agree with our old ratios. Their removal is caused by inherited corpus high_variance/replicate_spread notes retained after selecting a singleton pair. That flags historical heterogeneity; it does not demonstrate an error in the selected single-document pair. Consequently the planned ablation tests a data-selection change plus one alternate measurement, not a proven correction of four bad labels. Do not silently remove them from the original47 snapshot.

## Artifacts and verification scope

`experiments/results/glm-rebuild4-audit/` contains accepted.csv, audit.json, comparison.csv, omitted-four.csv, raw-checks.json and raw-records.json. The accepted schema preserves raw pair metadata, cells/strain, exposure flags, source-export hash and explicit protocol limitations. No new structure was discovered by this rebuild.

Read-only test command in `/Users/mike/Projects/vacv-si`:

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs /Users/mike/Projects/vacv-si-novel/.venv/bin/python -m pytest -p no:cacheprovider tests/test_external_v2_pairs.py -q
```

Result18 passed. No producer execution, network retrieval, main-tree edits, modeling, commits or submissions were performed by this audit agent. Main's pre-existing dirty state was preserved.
