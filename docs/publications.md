# Primary publications behind the competition rows

The competition groups its compounds by `strain` + `Culture`. Each distinct
string is one published assay table, and the `Title` column keeps that paper's
own compound label. Grouping on those two columns gives 50 groups over 917
compounds; the groups below are the ones whose compounds have no ChEMBL or
PubChem record, so a value for them has to come from the paper itself.

`data/processed/literature_manual.csv` holds every row extracted here, one line
per competition compound, with the paper, the paper's own label, the printed
numbers, the key used for the link and a confidence level.

## Papers whose activity table was read

| paper | identifier | competition compounds | table read from | train rows / identical numbers |
|---|---|---|---|---|
| Chiem et al., *Microbiol Spectr* 2023;11(4):e04745-22 | `10.1128/spectrum.04745-22` | 20 (19 distinct) | Tables 1-2 via Europe PMC JATS XML | 5 / 10 of 10 |
| Schormann et al., *Antimicrob Agents Chemother* 2011;55(11):5054-5062 | `10.1128/AAC.00278-11` | 23 | Table 1, open access | 7 / 14 of 14 |
| Moskalev et al., *Chemistry* 2025;7(2):34 | `10.3390/chemistry7020034` | 16 | Table 1, open access PDF | 4 / 5 of 6 |
| Prichard & Kern, *Viruses* 2010;2(9):1968-1983 | `10.3390/v2091968` | 6 | Table 1, open access | 3 / 6 of 6 |
| Priyamvada et al., *Front Microbiol* 2020;11:603 | `10.3389/fmicb.2020.00603` | 4 | Table 1, open access | 2 / 4 of 4 |
| Priyamvada et al., *Antiviral Res* 2021;191:105086 | `10.1016/j.antiviral.2021.105086` | 9 | EC50 column of the Fig. 2B heat map; no table in the paper | 4 / 4 of 4 (IC50 only) |
| Samolej et al., *Microbiol Spectr* 2024;12(4):e04072-23 | `10.1128/spectrum.04072-23` | 5 | values printed inside Fig. 1B and Fig. 4 panels | 2 / 2 of 2 (IC50 only) |
| Dsouza et al., *Antiviral Res* 2023;216:105651 | `10.1016/j.antiviral.2023.105651` | 3 | Results prose, no table | 1 / 2 of 2 |
| Nikitina et al., *Bioorg Med Chem Lett* 2023;79:129080 | `10.1016/j.bmcl.2022.129080` | 1 | Table 1, open access | 1 / 2 of 2 |
| Patent RU 2 800 448 C1 (NIOCH SB RAS / VECTOR) | `RU2800448C1` | 16 | Таблица 1, page images of the published patent | 7 / 13 of 14 |
| Revankar et al., *J Med Chem* 1984;27(11):1389-1396 | `10.1021/jm00377a002` | 6 | not read; values via ChEMBL doc `CHEMBL1122625` | 1 / 1 of 2 |
| Fan et al., *J Med Chem* 2006;49(14):4052-4054 | `10.1021/jm060404n` | 2 | not read; IC50 via ChEMBL doc `CHEMBL1138928` | 1 / 1 of 1 |

## Papers identified but not extractable

| paper | identifier | competition compounds | why not |
|---|---|---|---|
| Wu et al., *J Med Virol* 2019;91(11):2016-2024 | `10.1002/jmv.25544` | 26 (VACV Tiantan, Vero) | closed access; no copy in PMC, Unpaywall, OpenAlex or any repository. Only two SI values survive in the abstract (mycophenolate mofetil >334, tranilast >74), both of which match the competition rows and confirm the attribution |
| Sokolova, Kovaleva et al., *ChemMedChem* 2022;17(12):e202100771 | `10.1002/cmdc.202100771` | 43 (camphor/fenchone hydrazones and bornyl/fenchyl amides) | closed access behind Cloudflare; no repository copy. 16 of its rows were recovered from patent RU 2 800 448 C1 and 4 more from indexed body-text snippets |
| Nikitina et al., *Russ Chem Bull* 2019;68(3):634-637 | `10.1007/s11172-019-2467-6` | 17 (`PANI-15` … `PANI-54`) | Springer paywall; no OA location. The scaffold classes read off the SMILES partition exactly as the abstract describes, and the competition's blank IC50 cells fall on the N1-substituted analogues the abstract calls inactive, so the attribution is sound but no number was read |
| Sokolova et al., *Chem Biodivers* 2018;15:e1800153 | `10.1002/cbdv.201800153` | 12 (borneol/camphor esters, `Vaccinia virus` / Vero) | Wiley paywall. Attribution is on chemistry and author overlap only — treat as unconfirmed |
| Akishina et al., *Весці НАН Беларусі. Сер. хім. навук* 2023;59(3):211-224 | `10.29235/1561-8331-2023-59-3-211-224` | 1 (the second compound labelled `8` in the Moskalev group) | journal page unreachable |

## Group still unresolved

304 competition rows carry the Copenhagen/Grishak/ECTV strain strings with plain
numeric labels (`3a`, `4f`, `10`, …). 166 distinct labels over 304 rows, with
`7` appearing 9 times and `8` 8 times, so the bucket holds at least nine
separate papers and the label alone cannot identify which. Resolving it needs
scaffold clustering first, then one search per cluster. It was not pursued,
because the target turned out to be exactly computable from the columns the
competition already ships (see `docs/exact_target.md`), which leaves this work
no effect on the score.

## What the link keys are worth

Validation uses only the training rows, where `SI VACV` is known. A paper's own
compound label, matched inside a single `strain` + `Culture` group, reproduces
the printed numbers almost exactly:

| link key | train rows | CC50 identical | IC50 identical | SI within 0.3 log |
|---|---|---|---|---|
| paper label within the strain group | 24 | 23 of 23 | 24 of 26 | 18 of 19 |
| structure, for the patent surrogate | 7 | 7 of 7 | 6 of 7 | 7 of 7 |
| PubChem SID or ChEMBL id | 2 | 0 of 1 | 2 of 2 | 0 of 1 |

The label key is trustworthy. The three misses are all explained, and two of
them are errors in the competition file rather than in the extraction:

- **Moskalev compound 5** (train row 63): the competition pairs compound 5's
  CC50 with compound 4's EC50 (2.91 µM), one row up the column. Printed values
  are compound 4 → 75.50 / 2.91 and compound 5 → 299.40 / 75.53. The matching
  test row 16 (label 4) has the correct CC50 and a blank IC50.
- **`7b-ya-669-nav-1`** (train row 175): competition 0.59 µM, patent 0.6 ± 0.3.
  The competition SI is self-consistent with 0.59, so the journal table almost
  certainly prints 0.59 and the patent rounded.
- **Revankar CC50**: printed as a "toxic level" in µg/mL; the competition value
  is that number converted with the molecular weight.

The SID/ChEMBL key is the weak one, and for a reason worth recording: in the
Revankar rows the competition's `IC50_uM_VACV` is not an IC50 at all but the
dimensionless **virus rating** from ChEMBL assay `CHEMBL821893`, so the derived
SI has no meaning. Six test rows are affected.

## Other transcription errors found in the competition file

| row | compound | competition | printed |
|---|---|---|---|
| test 148 | Adefovir dipivoxil | IC50 0.038 µM | 0.302 µM; 0.038 is brincidofovir's value from the same sentence |
| test 230 | H42 | CC50 0.19 µM | 4.5 µM in the HeLa/VACV figure; 0.19 is the mpox/Hft value |
| test 512 | PAV-360 | IC50 1.97 µM | 1.98 µM; 1.97 is PAV-866 |
| train 205, test 248, test 570 | M23, M25, M24 | CC50 24 / 18 / 15 µM | those are the mpox/Hft values; the MRT compounds showed no HeLa cytotoxicity up to 40 µM |
| train 143 | M26 | CC50 30 µM | appears in no figure, table or sentence of the paper |
| train 0 … test 556 | PAV series | CC50 15.0 (10.0 for PAV-164) | the paper prints no CC50, only % cytotoxicity curves; these look digitised |
| test 170 / test 425 | Buparvaquone | the same table cell appears once in train and once in test | train/test leakage |
| test 425 | Buparvaquone | SMILES is not buparvaquone | — |
| train 87, train 158 | Albendazole, Leflunomide | SMILES are deuterated analogues | the papers tested the parent drugs |

Two strain strings are also wrong: the Schormann group's cowpox strain is
**Brighton**, not Grishak, and the Retro-1 analogues were assayed in **HeLa**,
not BSC40 (BSC-40 was used only for the entry, yield, plaque and confocal
assays).

## Caching

Raw downloads live under `data/raw/papers/` (gitignored): the Europe PMC XML for
Chiem, PDFs and page renders for Samolej, Priyamvada 2020, Moskalev and patent
RU 2 800 448 C1, the PAV figure images, and the NIOCH thesis and annual-report
PDFs used as negative controls.
