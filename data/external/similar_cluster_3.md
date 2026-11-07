# Public analogues of cluster_3: orthopox IC50/EC50 with paired CC50

File: `similar_cluster_3.csv`. Query: `data/problem/cluster_3.csv` (37 rows, 6 Murcko scaffolds).

Match rule: same Murcko scaffold as a query row, or Morgan r=2 (2048 bits) Tanimoto >= 0.5 to any query row.
`nearest_problem_row_index` is the `index` value of the closest query row (a same-scaffold row when one exists). `tanimoto` is the Tanimoto to that row.

## Columns

- `ic50_um` is the antiviral value in uM. It is an IC50 or an EC50; `source_id` says which (`IC50 activity ...` or `EC50 activity ...`). 64 rows are EC50, 11 are IC50.
- `cc50_um` is the CC50 in uM (nM divided by 1000).
- Relations are kept as ChEMBL records them: `ic50_relation` `=` 58, `>` 13, `<=` 4; `cc50_relation` `>` 64, `=` 11.
- `cell_line` is the CC50 cell line. The antiviral cell line is named in the ChEMBL assay description, whose ID is in `source_id`.
- `source_id` holds the ChEMBL activity IDs, the assay IDs and the molecule ID.
- A row is written only when the antiviral record and the CC50 record belong to the same ChEMBL document and the same molecule.

## Counts

| step | count |
|---|---|
| ChEMBL orthopox targets queried (vaccinia CHEMBL613493 and CHEMBL613168, cowpox CHEMBL612672, ectromelia CHEMBL5291666 and CHEMBL612728, monkeypox CHEMBL613120, orthopoxvirus CHEMBL4513207) | 7 |
| IC50 and EC50 activities on those targets | 2525 (IC50 662, EC50 1863) |
| distinct molecules with SMILES among them | 1733 |
| molecules matching the query rule | 59 (18 same scaffold, 41 Tanimoto >= 0.5 only) |
| antiviral records on matched molecules | 143 (IC50 41, EC50 102) |
| CC50 records on matched molecules | 41 (34 molecules) |
| TC50 records on matched molecules | 7, excluded because they are not CC50 |
| antiviral records paired with a CC50 in the same paper | 81 |
| paired records without a numeric antiviral value | 6, dropped |
| **rows in `similar_cluster_3.csv`** | **75** |
| distinct molecules in the output | 24 |
| distinct DOIs in the output | 7 |

Rows per query row (`nearest_problem_row_index`): 70 (tecovirimat) 65, 264 4, 30 3, 39 3.

Rows per virus (`virus_strain`): Vaccinia IHD-J 17, Cowpox 17, Cowpox CDV-resistant 14, Vaccinia Copenhagen 10, Monkeypox 7, Cowpox Brighton 3, Vaccinia WR 2, Cowpox Grishak 2, Ectromelia K-1 2, Ectromelia Moscow 1.

Rows per CC50 cell line: BSC-40 54, Vero 11, HFF 10.

## Output DOIs

| DOI | ChEMBL document | rows in CSV |
|---|---|---|
| 10.1021/jm061484y | CHEMBL1142975 | 54 |
| 10.1128/aac.00879-06 | CHEMBL1142200 | 3 |
| 10.1128/aac.00762-07 | CHEMBL1137648 | 7 |
| 10.1039/C8MD00442K | CHEMBL4368876 | 4 |
| 10.1039/d0md00108b | CHEMBL4699424 | 1 |
| 10.1016/j.bmcl.2022.129080 | CHEMBL5223146 | 3 |
| 10.1039/d4md00181h | CHEMBL6092128 | 3 |

## Gaps

- 62 antiviral records in 7 papers have no CC50 in the same paper and are not written:
  - 10.1021/acs.jmedchem.3c00069 (CHEMBL5346934): 26 records, 22 in ug/mL and 4 without a unit
  - 10.1016/j.ejmech.2008.08.010 (CHEMBL1158343): 14
  - 10.1039/C8MD00347E (CHEMBL4257560): 7
  - 10.1128/aac.01257-08 (CHEMBL1629620): 7
  - 10.1128/aac.01044-08 (CHEMBL1629642): 5
  - 10.1016/j.ejmech.2009.04.046 (CHEMBL1153622): 2
  - 10.1016/j.ejmech.2022.114742 (CHEMBL5331057): 1
- Nine of the unpaired molecules have a CC50 in another paper. Cross-paper pairs are not written.
- No ug/mL record was paired, so no ug/mL to uM conversion appears in the output (rule: value x 1000 / MW).
- Two CC50 records (CEMx174, MDCK) have no value or unit. Neither was paired.
- Matched molecules nearest to the imidazole (indices 38, 82, 120, 226, 266, 296, 421, 453), adamantyl (196), thiazolidine (140, 449) and camphor hydrazone (161, 164, 204, 349, 468) series have no paired CC50 row. Only indices 70, 264, 30 and 39 appear in the output.
- PubChem BioAssay: 85 unique SMILES (query rows and matched molecules) resolved to 72 CIDs. 64 CIDs appear in orthopox-named AIDs (582 AIDs). Fourteen of these AIDs report IC50, EC50 or CC50 for these CIDs. Every PubMed ID they cite maps to a ChEMBL document already used above, so PubChem added no pairs. The "Selectivity index" AIDs give ratios only and were not used.
- Europe PMC: 604 open-access hits for orthopox terms with CC50 or selectivity index. Chemotype searches (camphor, norbornene/tecovirimat, adamantyl, pyrrolopyrazinone) returned these papers, none of which is in ChEMBL by DOI:
  - 10.3390/molecules27154761 (PMC9331314, camphor iminothiazolidin-4-ones; IC50 and CC50 in Table 2)
  - 10.3390/v15010029 (PMC9865139, adamantane and bromantane amides; IC50 and TC50 in Tables 1 and 2)
  - 10.1002/cmdc.202100771
  - 10.1002/ardp.202100038
  - 10.1039/d6md00217j
  The first two were read in full text. Their tables use compound codes (2a to 3'd; 12a to 18b) and no SMILES, so the code-to-structure mapping was not made and no rows were written. The last three were not read.
