# Public analogues of cluster_0 with poxvirus IC50/EC50 and CC50

Output: `similar_cluster_0.csv`. One row per (IC50 or EC50 record, CC50 record) pair that
share a molecule and a ChEMBL document. Columns follow the spec.

## Method

1. ChEMBL assays whose organism or description names vaccinia, cowpox, ectromelia or
   monkeypox (`/assay.json`, `assay_organism__icontains`, `description__icontains`):
   1198 assays after filtering. Activities for them with standard type IC50 or EC50
   (`/activity.json`, `assay_chembl_id__in`): 3111 records, 2044 molecules.
   10 assays failed to download after per-assay retries and are not included
   (`failed_assays.json` is in the scratch directory, not in the repo).
2. Candidate molecules: Morgan radius 2, 2048 bits, RDKit. Kept if Tanimoto >= 0.5 to any
   of the 46 rows of `data/problem/cluster_0.csv`, or if the Murcko scaffold (no chirality)
   equals a problem scaffold. 184 molecules qualified (125 by Tanimoto, 172 by scaffold;
   some by both).
3. CC50: `/activity.json` with `molecule_chembl_id__in` and `standard_type__in=CC50` for the
   184 candidates. 587 records returned, no failures.
4. Pairing: an IC50/EC50 record is paired with every CC50 record of the same molecule and
   the same ChEMBL document. Pairs across different papers are not made.
5. Documents, assays and cell lines fetched from `/document.json`, `/assay.json`,
   `/cell_line.json`. Every document in the output has a DOI.
6. Spot check: activities 24978317 and 24978297 match the ChEMBL web API
   (73370 nM and 436260 nM, converted to 73.37 uM and 436.26 uM).

## Conversion

- nM to uM: divide by 1000. uM kept. mM multiplied by 1000.
- ug/mL to uM: value / parent MW (RDKit, MolWt of the ChEMBL parent SMILES) x 1000.
- Relations kept as ChEMBL reports them (`=`, `>`, `>=`). Records with no numeric value excluded.
- `ic50_um` holds EC50 values for 256 rows. `source` says `ChEMBL IC50` or `ChEMBL EC50`.
- `nearest_problem_row_index` is the problem row with the highest Tanimoto when that is
  >= 0.5. Otherwise it is the problem row with the same scaffold that has the highest
  Tanimoto. `tanimoto` is the value for that row, so scaffold-only rows can show values
  below 0.5 (minimum 0.354).

## Counts

| item | count |
|---|---|
| output rows | 336 |
| distinct molecules (SMILES) | 99 |
| distinct DOIs | 34 |
| IC50 rows / EC50 rows | 80 / 256 |
| ic50 relation `=` / `>` | 186 / 150 |
| cc50 relation `>` / `=` / `>=` | 192 / 142 / 2 |
| rows with a virus strain | 134 |
| rows with a cell line | 279 |
| rows where the CC50 assay cell line differs from the antiviral cell line | 219 |

Virus organism of the antiviral assay: vaccinia 247, cowpox 63, ectromelia 22, monkeypox 4.
The rows cover 22 of the 46 problem rows (`nearest_problem_row_index`).

## Excluded or missing (ChEMBL)

- 19 IC50/EC50 records with no numeric value.
- 136 IC50/EC50 records on candidate molecules with no CC50 record in the same document.
- 2 CC50 records with no value; 11 CC50 records with units that cannot be converted (no unit).
- 57 rows with no cell line in ChEMBL (assay has no cell link and no cell type).
- 202 rows with no strain in the assay record; 134 filled by matching a strain name that
  appears in the assay description.
- Only `standard_type` CC50 is used. MCC, TC50 and CC50 reported in other ways are not.
- CC50 and antiviral cell lines often differ. Read SI values with that in mind.

## PubChem and Europe PMC: not mined for values

- PubChem: 99 candidate molecules were mapped by InChIKey. 90 have at least one PubChem
  AID, and 4 have no record. The vaccinia/cowpox/ectromelia/monkeypox term search in
  pcassay is not specific (3200 AIDs for "vaccinia" alone), so the intersection is not a
  usable set of poxvirus assays. No values taken from PubChem.
- Europe PMC: a query for open-access papers with poxvirus, IC50 and CC50 gives 1172 hits.
  Full text was not parsed for tables. No values taken from Europe PMC. The 34 DOIs above
  all come from ChEMBL documents.

## Limits

- Coverage is ChEMBL only, as of the API snapshot at run time.
- The problem file `data/problem/cluster_0.csv` is untracked in git and was read as is.
- Values are not checked against the source papers. ChEMBL curation is taken as given.
