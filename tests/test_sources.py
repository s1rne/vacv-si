"""Source modules are tested against recorded payloads, not the live APIs.

A test that hits ChEMBL would be slow, flaky and useless in CI. What matters is
that the row builders turn a realistic payload into a conforming row, and that
the specific traps documented in `docs/data_sources.md` stay closed.
"""

import pandas as pd

from vacv.schema import SOURCE_COLUMNS, conform
from vacv.sources import ALL_SOURCES, OPEN_SOURCES, chembl

ACTIVITY = {
    "activity_id": 146764,
    "assay_chembl_id": "CHEMBL824281",
    "assay_description": "Effective dose required for antiviral activity against Vaccinia virus",
    "assay_type": "F",
    "bao_label": "organism-based format",
    "canonical_smiles": "Cn1cc(C(=O)c2ccc(N)cc2)cc1/C=C/C(=O)O",
    "document_chembl_id": "CHEMBL1124988",
    "document_year": 1990,
    "molecule_chembl_id": "CHEMBL95840",
    "pchembl_value": None,
    "standard_relation": "=",
    "standard_type": "ED50",
    "standard_units": "ug ml-1",
    "standard_value": "1000.0",
    "target_organism": "Orthopoxvirus vaccinia",
    "target_pref_name": "Vaccinia virus",
    "activity_comment": None,
    "data_validity_comment": None,
}

ASSAY = {
    "assay_chembl_id": "CHEMBL824281",
    "assay_organism": "Orthopoxvirus vaccinia",
    "assay_strain": None,
    "assay_type": "F",
    "cell_chembl_id": None,
    "description": "Effective dose required for antiviral activity against Vaccinia virus",
    "document_chembl_id": "CHEMBL1124988",
}

DOCUMENT = {
    "document_chembl_id": "CHEMBL1124988",
    "doc_type": "PUBLICATION",
    "doi": "10.1021/jm00172a026",
    "first_page": "2845",
    "journal": "J Med Chem",
    "src_id": 1,
    "title": "Synthesis and antimicrobial and cytotoxic activities of pyrrole-containing analogues.",
    "volume": "33",
    "year": 1990,
}


def test_every_registered_source_is_callable():
    assert set(OPEN_SOURCES) == {"chembl", "pubchem"}
    assert "kaggle" in ALL_SOURCES and "kaggle" not in OPEN_SOURCES
    for harvest in ALL_SOURCES.values():
        assert callable(harvest)


def test_chembl_row_conforms_and_keeps_provenance():
    row = chembl._row(ACTIVITY, {ASSAY["assay_chembl_id"]: ASSAY}, {DOCUMENT["document_chembl_id"]: DOCUMENT}, {})
    frame = conform(pd.DataFrame([row]))
    assert list(frame.columns) == SOURCE_COLUMNS
    assert frame.loc[0, "measurement_class"] == "antiviral"
    assert frame.loc[0, "doi"] == "10.1021/jm00172a026"
    assert frame.loc[0, "doi_provenance"] == "document"
    assert frame.loc[0, "document_id"] == "CHEMBL1124988"
    assert frame.loc[0, "source_id"] == "CHEMBL_ACT_146764"
    # ug/mL cannot be converted yet: the molecular weight is not known until
    # curation standardizes the structure.
    assert pd.isna(frame.loc[0, "value_um"])
    assert frame.loc[0, "data_validity"] == "mass_units_without_mw"
    assert frame.loc[0, "raw_units"] == "ug ml-1"


def test_chembl_row_drops_the_document_key_for_bulk_deposits():
    bulk = dict(DOCUMENT, doc_type="DATASET", doi=None, title="PubChem BioAssay data set", src_id=7)
    row = chembl._row(ACTIVITY, {ASSAY["assay_chembl_id"]: ASSAY}, {bulk["document_chembl_id"]: bulk}, {})
    assert row["document_id"] is None
    assert row["doi"] is None


def test_expandable_documents_excludes_depositor_containers():
    documents = {
        "CHEMBL1124988": DOCUMENT,
        "CHEMBL1201862": dict(DOCUMENT, document_chembl_id="CHEMBL1201862", doc_type="DATASET", src_id=7),
        "CHEMBL9999999": dict(DOCUMENT, document_chembl_id="CHEMBL9999999", doc_type="PUBLICATION", src_id=7),
    }
    assert chembl.expandable_documents(documents) == ["CHEMBL1124988"]


def test_biochemical_formats_are_recognised():
    assert chembl.is_biochemical("single protein format")
    assert not chembl.is_biochemical("organism-based format")
    assert not chembl.is_biochemical("cell-based format")
    assert not chembl.is_biochemical(None)


def test_cytotoxicity_row_from_a_cell_line_assay():
    activity = dict(
        ACTIVITY,
        activity_id=1,
        standard_type="CC50",
        standard_units="nM",
        standard_value="50000",
        assay_description="Cytotoxicity against uninfected Vero cells",
        target_organism="Chlorocebus aethiops",
        bao_label="cell-based format",
    )
    assay = dict(ASSAY, assay_type="T", cell_chembl_id="CHEMBL_CELL_VERO")
    cells = {"CHEMBL_CELL_VERO": {"cell_chembl_id": "CHEMBL_CELL_VERO", "cell_name": "Vero"}}
    row = chembl._row(activity, {assay["assay_chembl_id"]: assay}, {DOCUMENT["document_chembl_id"]: DOCUMENT}, cells)
    assert row["measurement_class"] == "cytotox"
    assert row["cell_line"] == "Vero"
    assert row["value_um"] == 50.0
