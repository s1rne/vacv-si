# Poxvirus analogues for the cluster_rest problem set

Output: `similar_rest.csv` (332 rows). Source: ChEMBL REST API only. PubChem and Europe PMC were not queried in this pass.

## Method

- Problem set: `data/problem/cluster_rest.csv`, 184 rows, 153 Murcko scaffolds. 31 scaffolds have 2 rows, 122 have 1, none more. All rows processed in one pass; grouping did not change the order of work.
- Assay discovery: `assay/search.json?q=` for vaccinia, cowpox, ectromelia, monkeypox, orthopox, mpox, variola. 1331 unique assays.
- Activities: IC50, EC50, CC50 from those assays (3233 records). 3078 IC50/EC50 records with a poxvirus name in the assay description, from 1991 molecules and 160 documents.
- CC50: all CC50 records from the same 160 documents (4373 records).
- Analogue test (RDKit, Morgan radius 2, 2048 bits, stereo ignored): Tanimoto >= 0.5 to any problem compound, or Murcko scaffold equal to a problem `scaf`. 275 of 2575 ChEMBL molecules passed.
- Pairing: an IC50/EC50 record is paired with a CC50 record of the same molecule and same document. If several CC50 records share the document, the cell line must match.
- Units: nM, uM, mM, pM, M, and ug/mL or ng/mL (divided by RDKit MW). Relations kept from ChEMBL.
- `source` is `ChEMBL IC50` or `ChEMBL EC50`. `ic50_um` holds the EC50 value in the 231 EC50 rows.
- `source_id` is `<antiviral activity_id>/<CC50 activity_id>`. `doi` is taken from ChEMBL `document.json`.
- `virus_strain` is the text from the assay description (for example "Vaccinia virus Copenhagen"). `cell_line` is the cell type of the CC50 assay.
- `nearest_problem_row_index` is the problem row with the highest Tanimoto (the problem `index` column, as an integer). `tanimoto` is that value, rounded to 3 decimals.

## Counts

| item | value |
|---|---|
| output rows | 332 |
| distinct DOIs | 42 |
| distinct analogue SMILES | 114 |
| problem rows that are nearest for at least one row | 40 of 184 |
| IC50 rows / EC50 rows | 101 / 231 |
| rows with Tanimoto >= 0.5 | 213 |
| rows with Tanimoto < 0.5 (scaffold match only) | 119 |
| rows with blank cell_line | 4 |
| IC50 relation: `=` / `>` / `<=` / `>=` | 259 / 66 / 5 / 2 |
| CC50 relation: `>` / `=` / `>=` | 217 / 114 / 1 |

## Gaps and caveats

- 119 rows match by scaffold but have Tanimoto < 0.5 to every problem compound. They are kept because the spec allows a scaffold match. Filter on `tanimoto >= 0.5` for close analogues only.
- Stereo is ignored in the Tanimoto. Example: row 0 is identical to problem row 0 apart from the stereocentre.
- Dropped at pairing: 252 matched IC50/EC50 records had no CC50 in the same document; 80 had several CC50 records with no unique cell-line match.
- Dropped for units: 56 records had no `standard_units`, and 2 EC50 records in ug.mL-1 were not converted.
- ChEMBL assay chunks that failed after retries (10 assays, not retrieved): CHEMBL679130, CHEMBL679134, CHEMBL676608, CHEMBL686574, CHEMBL686578, CHEMBL692744, CHEMBL692745, CHEMBL701504, CHEMBL699634, CHEMBL699645. Analogues measured only in these assays are missing.
- PubChem BioAssay and Europe PMC open-access full text were not searched. Coverage is ChEMBL-only.
- Strain and cell line are taken from free-text assay descriptions and may be incomplete.
- The problem set itself was not checked for stereo or salt forms against ChEMBL parent structures.
