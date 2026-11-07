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
| Nikitina et al., *Russ Chem Bull* 2019;68(3):634-637 | `10.1007/s11172-019-2467-6` | 17 (`PANI-15` … `PANI-54`; `PANI` is the author's own compound code) | Springer paywall; no OA location. The scaffold classes read off the SMILES partition exactly as the abstract describes, and the competition's blank IC50 cells fall on the N1-substituted analogues the abstract calls inactive, so the attribution is sound but no number was read |
| Sokolova et al., *Chem Biodivers* 2018;15:e1800153 | `10.1002/cbdv.201800153` | 12 (borneol/camphor esters, `Vaccinia virus` / Vero) | Wiley paywall. Attribution is on chemistry and author overlap only — treat as unconfirmed |
| Akishina et al., *Весці НАН Беларусі. Сер. хім. навук* 2023;59(3):211-224 | `10.29235/1561-8331-2023-59-3-211-224` | 1 (the second compound labelled `8` in the Moskalev group) | journal page unreachable |

## The Copenhagen / Grishak / ECTV bucket, resolved by chemotype

The 304 rows that carry the Copenhagen, Grishak and ECTV strain strings with
plain numeric labels are not one paper and not nine unknown ones. All of them
come from two Novosibirsk groups - NIOCH SB RAS (medicinal chemistry) and the
State Research Center of Virology and Biotechnology VECTOR (the virology) -
and the route into them is the institute's own patent and dissertation
listings rather than a literature search:

- `web3.nioch.nsc.ru/nioch/index.php/ru/nauka-v-niokh/publikatsii/patenty`
  is a complete index of NIOCH patents. Twelve of them name orthopoxviruses,
  and each one is a different chemotype in this bucket.
- `web3.nioch.nsc.ru/nioch/index.php/ru/obrazovanie-v-niokh/dissertatsionnyj-sovet/dissertatsii-i-avtoreferaty`
  carries full dissertation PDFs, not only the abstracts.
- Russian patent full text is reachable at
  `https://www1.fips.ru/registers-doc-view/fips_servlet?DB=RUPAT&DocNumber=<n>&TypeFile=html`
  (cp1251, tables inline for post-2015 filings and as page images before that).
  Google Patents and Espacenet both refuse automated requests; FIPS also caps
  the number of documents per session, so fetches have to be spaced out.

The competition's internal registry codes turn out to be the chemists'
initials: `as-` is A. S. Sokolova, `ks-` is K. S. Kovaleva, `ya-` is
O. I. Yarovaya. That is what ties a label to a laboratory and therefore to a
paper.

### Tables linked by `scripts/link_literature_tables.py`

Each table is transcribed verbatim into
`data/literature_tables/sources.json` and linked by the paper's own
compound label, restricted to the `strain` + `Culture` groups the table can
belong to and to the structural series it is about (a SMARTS filter per
series). Labels collide across papers in this bucket - `4a`, `6b` and `8b`
each name two different compounds inside the Copenhagen/Vero group - so the
series filter is what makes the label key safe.

| paper | identifier | table | train rows / CC50 identical / IC50 identical | test rows covered |
|---|---|---|---|---|
| Mozhaitsev et al., *Viruses* 2023;15(1):29 | `10.3390/v15010029` | Table 1, adamantane cage amides, 15 rows | 3 / 3 of 3 / 3 of 3 | 12 |
| Nikitina et al., *Bioorg Med Chem Lett* 2023;79:129080 | `10.1016/j.bmcl.2022.129080` | Table 1, 1-hydroxyimidazoles `1a-f`, `2a-l` | 3 / 3 of 3 / 3 of 3 | 7 |
| Kovaleva et al., *Molecules* 2022;27(15):4761 | `10.3390/molecules27154761` | Table 2, camphor thiosemicarbazones, thiazolidinones and thiazol-2-imines, 32 rows | 10 / 10 of 10 / 10 of 10 | 20 |
| Sokolova et al., *MedChemComm* 2018;9(11):1746 | `10.1039/c8md00347e` | Table 1, camphor thiazoles and thiazolidin-4-ones, 16 rows | 3 / 3 of 3 / 3 of 3 | 12 |
| Basanova et al., *RSC Med Chem* 2024;15(9):3196 | `10.1039/d4md00181h` | Table 1, 2-aryl-1-hydroxyimidazoles, 22 rows | 7 / 7 of 7 / 7 of 7 | 14 |
| Patent RU 2 794 763 C1 (NIOCH / VECTOR) | `RU2794763C1` | Table 1, `Ia-Id`, linked by molecular formula | 0 | 1 |

Every training row these six tables touch reproduces both printed numbers
exactly: **26 train rows, 26 of 26 CC50 identical, 26 of 26 IC50 identical.**
That is the same accuracy the label key showed before and it is now measured
on a sample four times larger.

Two findings worth keeping:

- **`10.1039/d4md00181h` is numbered one series lower in the competition
  file.** The paper's `5a-e`, `6a-b`, `7a-e`, `9a`/`9c`, `10a`/`10b`, `11a-e`
  and `13` are the file's `4a-e`, `5a-b`, `6a-e`, `8a`/`8c`, `9a`/`9b`,
  `10a-e` and `12`. The offset is identical on all 22 rows and exact on the
  seven training rows, so the file was built from an earlier version of that
  table (the SSRN preprint, `10.2139/ssrn.4619442`) rather than the published
  one. `paper_label` keeps the published label.
- **Selivanov's N-amidoimides print µg/mL, and the competition converted
  them.** `10.30906/0023-1134-2018-52-10-3-7` (diamond open access at
  `chem.folium.ru`) reports compounds `III-VIII` as 15.63/1.70, 6.42/1.05,
  >100/0.38, 18.62/0.95, >100/1.30 and >100/1.28 µg/mL. Dividing each pair by
  the molecular weight of the competition row labelled `3`, `4`, `5`, `6`, `7`
  and `8` in the Copenhagen/Vero group reproduces that row's µM values to
  better than 0.2% on all six, which both identifies the paper and explains
  the censored `100` that appears as 270.75, 33.64 and so on. Those six rows
  already had open-data values, so nothing was substituted.

### What still blocks the remaining 64 test rows

Test-row coverage inside the six biggest Copenhagen/Grishak/ECTV groups, counting
`competition_open_data_lookup.csv` and `literature_manual.csv` together:

| group | test rows | from open data | from literature | covered | gap |
|---|---|---|---|---|---|
| `Vaccinia virus (strain Copenhagen). culture Vero` | 84 | 51 | 36 | 64 | 20 |
| `… cowpox (strain_Grishak), ectromelia virus (ECTV, strain K-1)` | 46 | 4 | 11 | 15 | 31 |
| `… cowpox (strain_Grishak), ectromelia (strain K-1)` | 30 | 22 | 12 | 29 | 1 |
| `Vaccinia virus, lung fibroblast cells` | 13 | 13 | 0 | 13 | 0 |
| `Vaccinia virus (strain Copenhagen). culture Vero,` | 11 | 0 | 0 | 0 | 11 |
| `… cowpox virus (strain_Grishak)` | 25 | 15 | 17 | 24 | 1 |

Four paywalled papers account for all 64, and each is now identified to the
compound series:

| gap | rows | source | status |
|---|---|---|---|
| the whole ECTV/Grishak group's `2x-ks`, `3x-ks`, `7x-as`, `9x-qs`, `10x`, `11x` series | 31 | Sokolova, Kovaleva, Kuranov et al., *ChemMedChem* 2022;17(12):e202100771, `10.1002/cmdc.202100771` | bornylamine (`7`, `8`), fenchylamine (`9`, `10`), isobornyl ester (`11`) amides and camphor/fenchone acylhydrazones (`2`, `3`). Wiley refuses automated requests for the article, the PDF and the supporting information; OpenAlex reports no repository copy. Patent RU 2 800 448 C1 claims only 16 of them (`Ia-f`, `IIa-f` plus stereoisomers) and those 16 are already extracted. The NIOCH 2022 annual report shows the lead values as an image only |
| `3a`-`3p` N-aryl ureas | 9 | Selivanov et al., *Khim.-Farm. Zh.* 2017;51(6):13-17, Eng. *Pharm Chem J* 51(6):439-443, `10.1007/s11094-017-1629-9` | the 2018 companion paper in the same journal is diamond open access, this one is not online: `chem.folium.ru` publishes issues back to vol. 51 no. 11 only, and the 2017 no. 6 issue id is not served |
| bornyl and isobornyl amino-acid esters and amides `18`, `22`, `23`, `28`, `38`, `39` | 6 | Sokolova et al., *Arch Pharm* 2021;354:e2100038, `10.1002/ardp.202100038` | OpenAlex calls it bronze open access, but Wiley returns 403 for `pdfdirect`. Patent RU 2 687 254 C1 covers only three of the series (`Ia-Ic`) and prints its table as page images |
| `PANI-15` … `PANI-54` | 11 | Nikitina et al., *Russ Chem Bull* 2019;68(3):634-637, `10.1007/s11172-019-2467-6` | closed, no repository copy. `PANI` is the author's own code (P. A. Nikitina, MUCTR). Her group's later tables are open (`10.1016/j.bmcl.2022.129080`, `10.1039/d4md00181h`) but none reprints the 2019 numbers; neither does patent RU 2 794 763 C1, which covers the 4-nitrophenyl analogues instead |

Remaining leads not followed: the apocamphanyl/isocamphanyl patent listed on
the NIOCH page carries a wrong number there (RU 2 867 684 is a polymerisation
catalyst), and its own follow-up paper, *RSC Med Chem* 2026
`10.1039/d6md00217j`, renumbers the series (`13d`), so neither maps onto the
file's `9x`/`10x` labels; K. S. Kovaleva's dissertation
(`KovalevaKS_diss.pdf`, Table 3) prints the camphor acylhydrazone screen but
under dissertation numbering (`130a-g`, `131b-e`, `133g-s`), which can only be
linked through the competition's own concentration columns and so was not used.

## Group still unresolved

This section recorded the bucket as unresolved. It is now partly resolved -
see the section above, which identifies the chemotype behind each cluster, the
paper or patent behind each chemotype, and the four paywalled papers that hold
the 64 test rows still without a value.

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
