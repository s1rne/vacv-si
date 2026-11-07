# Public analogues of cluster_2: orthopox IC50/EC50 with same-document CC50

Output: `similar_cluster_2.csv` (71 rows). Source queried: ChEMBL REST API only.

## Query

| item | count |
|---|---|
| rows in `data/problem/cluster_2.csv` | 38 |
| unique parent structures (salts removed) | 35 |
| Murcko scaffolds (match the `scaf` column for all rows) | 6 |

## Candidate retrieval

| step | count |
|---|---|
| ChEMBL similarity search, threshold 50, per parent structure | 35 searches |
| unique ChEMBL molecules returned | 446 |
| passing Morgan r2 (2048 bit) Tanimoto >= 0.5 to a query, or same scaffold | 446 |
| candidates added only by same scaffold (Tanimoto < 0.5) | 0 |

Scaffold substructure search was not run. Candidates come from the similarity search only.

## Activities

| step | count |
|---|---|
| molecules with IC50, EC50 or CC50 records fetched | 446 |
| orthopox antiviral IC50/EC50 records (description names vaccinia, cowpox, ectromelia, monkeypox, variola or mpox) | 244 (87 molecules) |
| of those, molecules with a CC50 record in ChEMBL | 64 |
| of those, molecules with a CC50 in the same document | 35 |

## Output rows (71)

| attribute | value |
|---|---|
| molecules (distinct SMILES) | 35 |
| problem rows with at least one analogue | 12 of 38 |
| DOIs | 7 |
| cell lines | Vero 38, BSC-40 16, HFF 15, HEL 2 |
| virus | vaccinia Copenhagen 21, vaccinia IHD-J 16, cowpox 16, ectromelia 12, vaccinia WR 2, vaccinia (strain not stated) 2, monkeypox 1, variola 1 |
| IC50 relation | `=` 64, `<=` 5, `>` 2 |
| CC50 relation | `>` 61, `=` 9, `>=` 1 |

| DOI | rows |
|---|---|
| 10.1016/j.ejmech.2021.113485 | 37 |
| 10.1021/jm061484y | 16 |
| 10.1128/aac.00762-07 | 8 |
| 10.1016/j.bmcl.2011.04.126 | 4 |
| 10.1128/aac.00879-06 | 3 |
| 10.1021/jm0707166 | 2 |
| 10.1039/d0md00108b | 1 |

## Excluded records

| reason | count |
|---|---|
| antiviral record carries a ChEMBL data validity flag ("Outside typical range") | 22 |
| CC50 record carries a ChEMBL data validity flag | 3 (of the paired set) |
| same-document CC50 exists, but for a different cell line | 57 |
| antiviral record without a numeric value (relation empty) | 8 |
| molecules with orthopox antiviral data and no CC50 anywhere in ChEMBL | 23 |
| molecules with a CC50 in ChEMBL but no same-document pair | 29 |

## Method

- Pair rule: an antiviral IC50/EC50 record and a CC50 record for the same molecule in the same ChEMBL document, with the same cell line (`assay_cell_type`). The first matching CC50 is used.
- Units: nM / 1000 to uM. Mass units (ug/mL) converted with the ChEMBL `full_mwt` of the molecule record. This affects 2 output rows (CHEMBL392630, CHEMBL219724). If the tested form was a salt, the value is off by the salt mass ratio.
- `virus_strain` is the species plus the strain name matched in the assay description; blank strain shows the species only.
- `source_id` is `document_chembl_id:IC50_activity_id:CC50_activity_id`.
- `nearest_problem_row_index` is the problem row with the highest Tanimoto. Parent 9a occurs twice in the problem file (rows 35 and 518); the first occurrence is used.
- `smiles` is the ChEMBL canonical SMILES of the record, which can include a salt fragment.

## Gaps and caveats

- Not searched: PubChem BioAssay and Europe PMC open-access full text. Only ChEMBL was used. Literature-only values are absent.
- ChEMBL REST returned HTTP 500 for a share of requests during the run. All requests were retried; the final build reported 0 failed requests and all 446 molecules have activity data.
- CC50 values with `>` are upper bounds. The SI derived from them is a lower bound.
- Outliers kept as recorded by ChEMBL: CHEMBL485653 CC50 = 1276 uM (doc 10.1039/d0md00108b) gives an SI of about 1.3e5 against IC50 0.01 uM.
- 26 of 38 problem rows have no paired public record in this search.
