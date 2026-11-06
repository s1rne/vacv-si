# Open data sources for VACV selectivity index

Every number in `data/processed/` comes from one of the sources marked
**used** below. Sources marked **evaluated** were checked against their live
endpoints and rejected for the stated reason; they are listed so that the
rejection is auditable rather than implied.

Reproduce everything with:

```sh
uv sync
uv run python -m vacv.cli all      # fetch -> curate -> build -> validate
uv run pytest
```

Raw API responses are cached under `data/raw/<source>/` (gitignored) keyed by a
hash of the request URL, so a second run is offline and produces the same
bytes. Add `--refresh` to re-query.

---

## Used

### 1. ChEMBL (primary source)

| | |
|---|---|
| endpoint | `https://www.ebi.ac.uk/chembl/api/data` (REST, JSON, no key) |
| licence | data **CC BY-SA 3.0**; EBI terms of use. Share-alike propagates to this dataset |
| module | `vacv/sources/chembl.py` |
| command | `uv run python -m vacv.cli fetch --source chembl` |

Queries, in the order the module issues them:

1. `GET /target.json?organism__icontains=vaccinia&limit=1000`
   → **8 targets**: `CHEMBL613493` (Orthopoxvirus vaccinia, ORGANISM),
   `CHEMBL613168` (Vaccinia virus WR, ORGANISM), and six SINGLE PROTEIN
   targets (strain WR, Copenhagen).
2. `GET /assay/search.json?q=vaccinia&limit=100&offset=…`
   → **832 assays** whose free text mentions vaccinia. This is the step that
   reaches cytotoxicity assays: those are filed against a *cell-line* target,
   so a virus-target query never sees them.
3. `GET /activity.json?target_chembl_id=…` for each target, and
   `GET /activity.json?assay_chembl_id__in=…` for the 832 assays, in batches of
   25 → **56 801 activities tied to vaccinia directly**.
4. `GET /document.json?document_chembl_id__in=…` → **353 documents**, of which
   **347** are real publications (`doc_type == "PUBLICATION"`, `src_id == 1`)
   and carry a DOI.
5. `GET /activity.json?document_chembl_id__in=…` for those 347 publications
   → **134 383 activities** in total. This widening is what makes
   same-publication pairing possible: a paper that reports "ED50 against
   vaccinia virus" also reports "CC50 in Vero cells", and only the first is
   reachable from the virus target.
6. `GET /assay.json?assay_chembl_id__in=…` → `cell_chembl_id`, `assay_strain`,
   `assay_type`, BAO format.
7. `GET /cell_line.json?cell_chembl_id__in=…` → cell line names.

#### Caveats

- **`CHEMBL1201862` is a trap.** ChEMBL's "document" is not always a paper.
  `CHEMBL1201862` is titled *PubChem BioAssay data set*, has
  `doc_type == "DATASET"`, and owns **7 434 992** activities from every
  depositor PubChem ever had. An unguarded document widening downloads most of
  ChEMBL - the first run of this harvester pulled 156 000 rows and 180 MB
  before it was stopped. Worse than the download: treating that container as
  "the same publication" would pair a vaccinia IC50 with a cytotoxicity value
  from an unrelated deposit, which is precisely what the strict pairing rule
  exists to prevent. The module therefore expands only
  `doc_type == "PUBLICATION"`, caps any single document at
  `MAX_ACTIVITIES_PER_DOCUMENT = 20 000`, and leaves `document_id` empty for
  non-publication documents so they can never act as a pairing key.
- **The endpoint name does not tell you what was measured.** Host-cell
  cytotoxicity is filed under `CC50`, `TC50`, `CD50`, `MCC`, `GI50` *and*
  `IC50`. `vacv/classify.py` decides from the endpoint name plus the assay free
  text, and refuses (`measurement_class == "unknown"`) rather than guessing.
- **Mass concentrations.** Many pre-2000 records are in `ug.mL-1`; ChEMBL does
  not compute `pchembl_value` for them. Converting needs a molecular weight, so
  `vacv/curate.py` does the conversion *after* RDKit standardization, using the
  parent structure's weight.
- **Biochemical assays are excluded from pairing.** An IC50 against purified
  vaccinia D1 methyltransferase is a real measurement and a meaningless half of
  a selectivity index. Rows whose BAO format is a protein/nucleic-acid format
  are kept in the measurement table and filtered out by
  `build_dataset.antiviral_side`.
- `activity_comment` and `data_validity_comment` carry source-side warnings
  ("Not Active", "Outside typical range"); both are preserved in the
  measurement table.

### 2. PubChem BioAssay

| | |
|---|---|
| endpoints | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils` (db `pcassay`), `https://pubchem.ncbi.nlm.nih.gov/rest/pug` |
| licence | PubChem imposes no restrictions on use or redistribution; individual depositor terms may differ |
| module | `vacv/sources/pubchem.py` |
| command | `uv run python -m vacv.cli fetch --source pubchem` |

Query: `esearch.fcgi?db=pcassay&term=vaccinia[ANAM] OR vaccinia[TXNM]`
→ **844 assays**.

- **The naive query is unusable.** `term=vaccinia` over all assay text returns
  **3200** assays, and the majority are false positives: modified vaccinia
  Ankara expressing T7 polymerase is a standard *tool* in Ebola and RSV
  minigenome assays, so "vaccinia" appears in protocols that measure nothing
  about poxviruses. AID 1806002, the first hit, is an Ebola assay. Restricting
  to the indexed assay-name (`ANAM`) and taxonomy (`TXNM`) fields fixes this.
- **827 of the 844 are ChEMBL re-deposits.** Depositor breakdown: ChEMBL 827,
  Broad Institute 5, Shanghai Institute of Organic Chemistry 4, BindingDB 3,
  NCATS 3, NCATS/FGL 1, ICCB-Longwood 1. Harvesting the ChEMBL mirror here
  would duplicate source 1 with less metadata (no DOI, no cell-line
  dictionary), so those are skipped and the count is reported rather than
  hidden. Pass `include_chembl_deposits=True` to override.
- Extending the search to the whole orthopoxvirus family
  (`cowpox|monkeypox|variola|orthopoxvirus` in `ANAM`/`TXNM`) gives **975**
  assays with the **same** non-ChEMBL depositor set. PubChem holds no
  additional poxvirus screening programme.
- **13 assays are used**, yielding **3555 measurements**, all on the antiviral
  side (AC50 3531, IC50 23, EC50 1) and **zero cytotoxicity**. The bulk is the
  NCATS qHTS poxvirus screen (AID 720579 mCherry reporter, AID 720580 Venus
  reporter), whose per-compound `Phenotype` column is `Inhibitor` for every
  active substance; the cytotoxic compounds were scored inactive and carry no
  AC50. So PubChem contributes IC50 halves that can only be paired at the
  weakest `compound` level.
- **Two PUG REST quirks.** The `sid=` filter is honoured by
  `/assay/aid/<aid>/CSV` and silently *ignored* by
  `/assay/aid/<aid>/concise/JSON` - requesting two SIDs from AID 720579 via
  `concise` downloads 67 MB. And every CSV table starts with `RESULT_TYPE` and
  `RESULT_DESCR` pseudo-rows that carry column metadata, not measurements;
  both have an empty SID, which is the test the module uses.
- **No DOI is emitted for PubChem rows.** An assay's `xref` PMID is a citation
  the depositor attached, not the publication of the measurement: AID 720579
  cites a monkeypox *epidemiology* paper (`10.3201/eid1306.061540`) that
  contains no chemistry. Those citations go into `reference`, and `doi` is left
  empty rather than filled with something misleading. `doi_provenance`
  distinguishes `document` (ChEMBL, the DOI of the paper the activity belongs
  to) from `assay_citation`.

### 3. The organisers' training split (optional, local, not an open source)

`vacv/sources/kaggle_competition.py` reads `data/raw/kaggle/*train*.csv` when
present. It is **off by default** (`--include-competition` turns it on) and the
glob deliberately excludes `test.csv`, whose labels are what the competition
asks us to predict. Everything in `data/processed/` is built without it.

---

## Evaluated and not used

Each was checked against its live endpoint. "No VACV cell-based data" means the
resource contains no cell-based vaccinia potency or host-cell cytotoxicity, not
merely that it was hard to query.

| source | reachable | VACV cell-based IC50/CC50 | licence | why not used |
|---|---|---|---|---|
| **BindingDB** | yes | IC50 only, never CC50 | curated data CC BY 3.0 US (site) / CC BY 4.0 (2024 NAR paper); ChEMBL-imported subset stays CC BY-SA 3.0 | Schema stores ligand-target affinity, so no cytotoxicity exists in it. Its only orthopox content is 19 IC50 records against VACV F13L/p37 (UniProt P04021, all from `10.1016/j.ejmech.2021.113485`) plus 3 patent assays, and all of it is already mirrored into PubChem as BindingDB-sourced AIDs, which step 2 above sees. |
| **PDBbind / PDBbind-CN** | partly (`pdbbind-plus.org.cn` is a JS shell; the old site is dead) | no | **redistribution forbidden without permission** | Protein-ligand affinity only. An RCSB structured search for `source organism = Vaccinia virus` AND `rcsb_binding_affinity exists` returns **zero** entries (controls: vaccinia alone 231, human + affinity 6826). The four "vaccinia" PubChem AIDs from this depositor (1811, 977608/10/11) are text hits on a generic PDBbind dump. |
| **Broad Drug Repurposing Hub** | yes | no | **non-commercial use only** (stated in the file header) | `repurposing_drugs_*.txt` and `repurposing_samples_*.txt` contain `pert_iname, clinical_phase, moa, target, smiles, InChIKey, pubchem_cid` and **no potency column of any kind**. Useful later for annotation, not as an activity source. |
| **DrugBank** | partly (Cloudflare 403 to scripts) | no evidence | main datasets **CC BY-NC 4.0**; only Vocabulary and Structures are CC0, and those hold no activity values | Non-commercial licence plus "All Academic DrugBank dataset downloads are temporarily paused" on the releases page. Not redistributable as part of a dataset. |
| **Guide to PHARMACOLOGY (IUPHAR/BPS)** | bulk CSVs open; API now needs a key | **no** | DB ODbL, contents CC BY-SA 4.0 | Across 24 970 interaction rows of version 2026.3.1: vaccinia 0, poxvirus 0, monkeypox 0, variola 0, cowpox 0, tecovirimat 0. Cidofovir and brincidofovir exist as ligands with zero interaction rows. A genuine content gap - the same file has 238 SARS-CoV-2 rows. |
| **Open Targets Platform** | yes (GraphQL v4, API 26.9.0) | no | CC0 1.0 | `search(queryString:"vaccinia")` returns only human **VRK1/VRK2** (vaccinia-*related* kinases). No viral targets by construction. |
| **ChEMBL-NTD deposited sets** | yes (`ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLNTD/`) | no | EBI terms, redistribution encouraged; some sets CC0 | All 27 sets are antiparasitic plus one Fraunhofer SARS-CoV-2 set. `grep -i 'pox|vaccinia|variola'` over them returns nothing. |
| **"ViralChEMBL" / a poxvirus ChEMBL subset** | n/a | n/a | n/a | Does not exist. ChEMBL's 67 sources include `SARS_COV_2` (src 52) and `ASAP` (src 71) but no poxvirus-specific source; there is no poxvirus analogue of the COVID-19 subset. |
| **DrugVirus.info 2.0** | yes | values exist but are **not machine-readable** | article CC BY 4.0; no portal data licence | `http://drugvirus.info/DrugVirus.csv` (350 KB, 1518 rows) has `Drug, Virus, Cell lines, Reference, …` and **no EC50/CC50/SI columns** - activity is an asterisk flag. 23 Poxviridae rows (VACV, CPXV, MCV) with PMIDs. The site serves its SI/EC50 values as **PNG images** (`./SIEC50/<Drug>.png`). Usable as a seed PMID list only. |
| **NCATS OpenData Portal** | **no** - TLS handshake fails (`SSL_ERROR_SYSCALL`, HTTP 000) for `/`, `/api/v1/datasets`, `/covid19/` | unverified | unverified | Unreachable. Its poxvirus content, the NCATS qHTS screens, is in PubChem (AID 720579/720580) and is already harvested by source 2. |
| **ASAP Discovery / COVID Moonshot** | yes | no | public domain, Zenodo `10.5281/zenodo.12191567` | Funded viral families are Coronaviridae, Flaviviridae, Picornaviridae, Influenza, Togaviridae - **Poxviridae is absent**, and its ChEMBL deposition (src 71, 181 assays) has zero pox hits. |
| **NIAID antiviral screening programme** | **no** - `niaid.nih.gov` returns a WAF human-verification page (HTTP 405) to scripts; `preclinical.niaid.nih.gov` is NXDOMAIN | no public database found | n/a | The historic Smee / Prichard / Kern / Quenelle screening results exist only in journal tables. Flagged as "could not be read", not "proven absent". |
| **BV-BRC** | yes (API v1.9.3) | no | BV-BRC public | Of 35 data types only `antibiotics` (97 records) and `genome_amr` are compound-related; `keyword(tecovirimat|cidofovir|brincidofovir)` and `genome_amr/?keyword(vaccinia)` all return `[]`. |
| **Therapeutic Target Database** | yes | no | no data licence page found | Live target index (1974 entries) has 25 virus-named targets (CoV, CMV, DENV, EBOV, EBV, HIV, HPV, MARV, herpes) and **zero** poxvirus/vaccinia/variola targets, so its tecovirimat and cidofovir records have nothing to key to. Legacy download URLs serve HTML, not files. |
| **ImmPort** | yes | no | ImmPort DUA | 23 vaccinia studies, all human/NHP **immunology** (SDY614, SDY28, SDY35). No compound screening. |
| **AntiviralDB** (mBio 2025, `10.1128/mbio.02013-25`) | shell loads, app does not (its asset host `img.haoranvip.com` fails TLS) | unclear | article CC BY; no data licence stated | The paper's full text contains **zero** occurrences of "EC50" or "CC50", and the site documents no bulk download or API. |
| **MPXV-Pred / BioinformaticsIASBS/MonkeyPox** | yes | no | **no licence file** | Contents are a binary drug-virus interaction matrix and similarity matrices; no EC50/CC50. Its paper (`10.1371/journal.pone.0299342`) mentions vaccinia zero times. |
| **Virus-CKB 2.0** | **no** - `cbligand.org` SERVFAIL on three resolvers | n/a | n/a | Dead resource, still cited in 2025 reviews. |
| **Open Source Antivirals / Antibiotics / Malaria** | `opensourceantivirals.org` and `opensourceantibiotics.org` are NXDOMAIN; `opensourcemalaria.org` now serves a gambling site | no | n/a | GitHub organisations exist but hold antimalarial and antibacterial data only. |
| **WHO mpox materials** | fact sheet yes | no dataset | WHO copyright, not open data | The fact sheet contains zero occurrences of `tecovirimat|brincidofovir|cidofovir|EC50|IC50`. |
| **IDAAPM** | yes over HTTPS | no evidence | not stated | FDA-approved-drug / ADMET / target resource, structurally target-centric. |

### Identified but not implemented: Europe PMC full-text tables

This is the one remaining route with real additional VACV data, and it is
documented here rather than implemented because it needs per-paper table
parsing that cannot be validated inside the time this data layer had.

- `GET https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=…&format=json&resultType=core`
  with `(vaccinia AND "CC50" AND "EC50") AND OPEN_ACCESS:Y AND HAS_SUPPL:Y`
  returns **163** papers; the orthopox-wide variant with `"selectivity index"`
  returns **138**. `resultType=core` includes `doi` and a per-record `license`
  field, so filtering to `cc by` / `cc0` is mechanical.
- `GET …/rest/<PMCID>/fullTextXML` returns the EC50/CC50/SI tables as
  `<table-wrap>` XML, not PDF. Verified on `PMC12846358`
  (`10.3390/v18010071`, CC BY): a table captioned *"Cytotoxicity and antiviral
  activity of selected benzimidazoles against Vaccinia virus expressed as CC50
  and EC50"* with columns `Vero-76 (CC50) | VV (EC50)` and rows such as
  `>100 | 14.4 ± 3.7 (>7)`.
- `GET …/rest/<PMCID>/supplementaryFiles` returns a zip. Use the **PMCID**;
  the plain PMID 404s.
- The NIAID Data Ecosystem API
  (`https://api.data.niaid.nih.gov/v1/query`) is a useful second pass for PLOS
  figshare table objects that carry their own CC BY 4.0, e.g.
  `10.1371/journal.pone.0055808.t003` at
  `https://ndownloader.figshare.com/files/487894` - VACV WR / Copenhagen /
  Lister EC50 and SI for cidofovir, CMX001, ST-246 and (S)-HPMP-5-azaC, 5.6 kB,
  no key, verified downloadable. Validate every `contentUrl` before counting
  it: `https://ndownloader.figshare.com/files/41521060` is a dead record.

The cost is per-paper heterogeneity: units, footnoted censoring (`>100`),
substituent-only compound identities that need the SAR scheme to resolve into a
SMILES, and SI values the authors computed from numbers they did not print.
None of that is safe to automate without eyeballing each paper, so the honest
move is to document the route and the specific verified entry points.

---

## Licence of the assembled dataset

The corpus is ChEMBL-dominated and ChEMBL is **CC BY-SA 3.0**, so share-alike
propagates: `data/processed/vacv_si.csv` must be redistributed under CC BY-SA
with attribution to ChEMBL (EMBL-EBI) and PubChem (NCBI). No PDBbind, DrugBank
or Drug Repurposing Hub content is included, so none of their restrictions
apply.
