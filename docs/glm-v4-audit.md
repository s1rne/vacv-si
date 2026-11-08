# GLM external_v2 audit at main f20d443

Decision: **no newly verified rows approved for ingestion this turn**. Do not load the142-row `measured_clean.csv` wholesale. This is a provenance failure, not a competition-overlap failure. The main working tree was dirty and was not modified; the producer was not run. The observations below describe files actually present, whose output hashes match their manifest.

## What is verified

- Independent structural recomputation using only competition SMILES found zero exact-SMILES and key14 intersections across measured142, censored851 and quarantine162. No stored-key mismatch or failed standardization in these exports.
- All142 measured rows are legacy tierA. There are71 document-level,69 strict,2 compound-level records;28 contain `aggregate_match`,7 contain multiple cell names. Thus the report's claim that all142 are conservative record-level exact pairs is unsupported.
-40 measured identities overlap our existing47 exact pairs, and all40 SI values agree. There are102 nominally new identities. This does **not** mean all102 are incorrect: they have not passed the required reconstruction.
- Among140 measured rows joined to the unit audit, the cytotoxic endpoint types are CC50:60, ED50:30, TC50:25, MNTD|TD50:14, CC50|IC50:6, MTC:4, IC50:1. Two compound-level rows lack a matching single-document audit. Strict CC50 semantics cannot be inferred from the export column name `cc50_um`.
- The focused helper tests pass:9 tests. They do not exercise the legacy producer-to-export path that retains these rows.

## Concrete failure chain

`experiments/external_regression_v2.py:_verify_half` accepts an aggregate geometric median as `aggregate_match`. `audit_record_units` considers both record_match and aggregate_match verified. `base_candidates` explicitly retains the old corpus SI and halves, adds these verification notes, and exports legacy candidates. `QUARANTINE_FLAGS` does not reject aggregate_match; the final exact/unflagged mask consequently admits them.

For example, AAZNZTJNDUZPDY has cell `HEL|HeLa`, CC50=0.7148426400264607, four cytotoxic measurements, and note `unit_verified_cc50:aggregate_match:14597502|14597630|14597634|14597640`. ACAAKOPRVNQCEH has CC50=2.97758291236, four measurements and another aggregate_match. They are not single-record pairs merely because their aggregates reproduce the legacy value.

The new conservative pairing helper exists, but it does not replace legacy tierA ingestion. New ChEMBL expansion reports zero conservative pairs in this manifest. Helper correctness therefore does not establish that measured_clean is built exclusively from those helpers.

## The only new strict singleton candidate fails actual assay conditions

Among nominally new rows, requiring strict pairing, one measurement per half, record_match on both halves, antiviral IC50/EC50 and cytotoxic CC50 leaves one candidate: MWOIGQWTBZSCGB, CHEMBL4052722, reported SI2.6315789473684212.

- Activity18262277 / assayCHEMBL4056706: IC50=19000nM, Vaccinia LIVP, CV1 cells, **4h preincubation followed by measurement48h after infection**.
- Activity18262178 / assayCHEMBL4056694: CC50=50000nM, CV1 cells, **4h preincubation followed by measurement24h later**.

The unit audit records time sets `[4,48]` and `[4,24]`. Its set-intersection condition accepts their common4h, confusing preincubation with assay endpoint time. Our older source-pair quarantine correctly records `different_exposure_times` for this identity. Cached original ChEMBL records were read directly; descriptions and cache hashes are saved in `experiments/results/glm-v4-audit/raw-record-evidence.json`.

## NIOCH-14 values are different provenance claims

For YDHJUPUAMAWEOT, our157026.3157894737 is the single-document pair CHEMBL5223146: activity24978328 IC50=7.6nM and activity24978308 CC50=1193400nM, both Vero and4-day measurements. Ratio1193.4/0.0076 reproduces our value. Full protocol equivalence remains a limitation, as already recorded in our47-row snapshot.

The GLM quarantine's153050.32067230286 uses IC50=0.0077974354758471726 and CC50=1193.4, with combined source `CHEMBL5223146|CHEMBL6092128`; notes include `pair_spans_documents` and `si_outside_window`. It is not a corrected version of our single-document pair. Do not replace either with the other's provenance or silently cap high SI.

## Required producer fix for GLM

1. Make conservative record-pair reconstruction the **sole numeric export source** across legacyA/B and expansionD/E. Existing tables may nominate documents/structures, but their SI, aggregated halves and `aggregate_match` cannot become labels. Emit raw activity/assay/document IDs, endpoints, relations, values, units, cells, strains and exposure evidence for each accepted pair.
2. Enforce antiviral IC50/EC50 × cytotoxic CC50 in every branch. Keep MCC, TC50, ED50, MTC, MNTD and generic IC50 cytotoxic records separately; do not rename or aggregate them into CC50. Preserve interval direction; two censored halves must not become an exact ratio.
3. Parse exposure by phase or conservatively reject ambiguous multi-time descriptions. Shared preincubation does not establish equal endpoint exposure. Unknown conditions remain explicit and cannot pass a known-equivalence claim.
4. Add end-to-end producer/export regressions: a legacy aggregate that reproduces its old value must still be rejected; cross-cell and mixed-endpoint aggregates must not enter measured_clean; the18262277/18262178 pair must be rejected; source-document and activity IDs must reproduce each emitted ratio; full train/test key14 and exact-SMILES exclusions must remain active.
5. Rebuild the exports under a new hash/version, report genuinely new record-supported identities against our47, and retain explicit quarantines. Do not relabel the existing142 as fixed after changing only helper tests.

## Audit artifacts and verification

`experiments/results/glm-v4-audit/` contains `audit.json`, joined measured/audit evidence, the40-row comparison, raw-record evidence and an empty `approved-new-subset.csv`. Empty means **zero newly verified/approved**, not proof that no valid new external pair exists anywhere.

Command run read-only in the main repository:

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs /Users/mike/Projects/vacv-si-novel/.venv/bin/python -m pytest -p no:cacheprovider tests/test_external_v2_pairs.py -q
```

Result:9 passed. No model fitting, producer execution, submission or main-repository mutation was performed for this audit.
