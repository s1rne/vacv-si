import pandas as pd
import pytest

from vacv.build_dataset import REQUIRED_COLUMNS, antiviral_side, build, cytotox_side, order_columns
from vacv.curate import curate
from vacv.schema import conform
from vacv.validate import errors, validate

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"


def _measurement(**overrides):
    row = {
        "smiles": ASPIRIN,
        "standard_type": "IC50",
        "measurement_class": "antiviral",
        "value_um": 1.0,
        "relation": "=",
        "raw_value": 1.0,
        "raw_units": "uM",
        "assay_description": "Antiviral activity against vaccinia virus in Vero cells",
        "assay_format": "cell-based format",
        "organism": "Vaccinia virus",
        "cell_line": "Vero",
        "source_db": "test",
        "source_id": "x",
        "document_id": "DOC1",
        "doi": "10.1000/test",
    }
    row.update(overrides)
    return row


def _curated(rows):
    kept, _ = curate(conform(pd.DataFrame(rows)), canonical_tautomer=False)
    return kept


def test_antiviral_side_requires_vaccinia_evidence():
    frame = _curated(
        [
            _measurement(source_id="a"),
            _measurement(
                source_id="b",
                assay_description="Antiviral activity against herpes simplex virus 1",
                organism="Human herpesvirus 1",
            ),
        ]
    )
    assert set(antiviral_side(frame)["source_id"]) == {"a"}


def test_antiviral_side_rejects_biochemical_assays():
    frame = _curated(
        [
            _measurement(source_id="cell"),
            _measurement(
                source_id="enzyme",
                assay_format="single protein format",
                assay_description="Inhibition of vaccinia virus D1 methyltransferase",
            ),
        ]
    )
    assert set(antiviral_side(frame)["source_id"]) == {"cell"}


def test_cytotox_side_does_not_require_vaccinia():
    frame = _curated(
        [
            _measurement(
                source_id="tox",
                standard_type="CC50",
                measurement_class="cytotox",
                assay_description="Cytotoxicity against uninfected Vero cells",
                organism="Homo sapiens",
                value_um=50.0,
            )
        ]
    )
    assert set(cytotox_side(frame)["source_id"]) == {"tox"}


def _pairable(smiles, document, cell, ic50, cc50, **extra):
    return [
        _measurement(
            smiles=smiles,
            source_id=f"{smiles}-{document}-ic",
            document_id=document,
            cell_line=cell,
            value_um=ic50,
            **extra,
        ),
        _measurement(
            smiles=smiles,
            source_id=f"{smiles}-{document}-cc",
            document_id=document,
            cell_line=cell,
            standard_type="CC50",
            measurement_class="cytotox",
            assay_description="Cytotoxicity against uninfected Vero cells",
            value_um=cc50,
        ),
    ]


def test_strict_pairing_requires_same_document_and_cell_line():
    frame = _curated(
        _pairable(ASPIRIN, "DOC1", "Vero", 1.0, 100.0)
        + [
            _measurement(smiles=CAFFEINE, source_id="c-ic", document_id="DOC2", cell_line="Vero"),
            _measurement(
                smiles=CAFFEINE,
                source_id="c-cc",
                document_id="DOC3",
                cell_line="HeLa",
                standard_type="CC50",
                measurement_class="cytotox",
                assay_description="Cytotoxicity against uninfected HeLa cells",
                value_um=10.0,
            ),
        ]
    )
    tables = build(frame)
    assert len(tables["strict"]) == 1
    assert tables["strict"].loc[0, "si"] == pytest.approx(100.0)
    # The caffeine halves come from different papers and different cells, so
    # only the weakest level may join them.
    assert len(tables["document"]) == 1
    assert len(tables["compound"]) == 2
    assert set(tables["relaxed"]["pairing_level"]) == {"strict", "compound"}


def test_relaxed_table_prefers_the_most_specific_level():
    frame = _curated(_pairable(ASPIRIN, "DOC1", "Vero", 2.0, 20.0))
    tables = build(frame)
    relaxed = tables["relaxed"]
    assert len(relaxed) == 1
    assert relaxed.loc[0, "pairing_level"] == "strict"
    assert relaxed.loc[0, "si"] == pytest.approx(10.0)


def test_si_is_the_ratio_and_passes_validation():
    frame = _curated(
        _pairable(ASPIRIN, "DOC1", "Vero", 1.0, 50.0)
        + _pairable(IBUPROFEN, "DOC2", "HeLa", 4.0, 2.0)
    )
    tables = build(frame)
    relaxed = order_columns(tables["relaxed"])
    assert list(relaxed.columns[: len(REQUIRED_COLUMNS)]) == REQUIRED_COLUMNS
    assert not errors(validate(relaxed))
    ratios = (relaxed["cc50"] / relaxed["ic50"]).tolist()
    assert relaxed["si"].tolist() == pytest.approx(ratios)
    assert sorted(round(x, 6) for x in ratios) == [0.5, 50.0]


def test_censored_pair_is_flagged_not_dropped():
    frame = _curated(
        [
            _measurement(source_id="ic", value_um=1.0),
            _measurement(
                source_id="cc",
                standard_type="CC50",
                measurement_class="cytotox",
                assay_description="Cytotoxicity against uninfected Vero cells",
                value_um=100.0,
                relation=">",
            ),
        ]
    )
    tables = build(frame)
    row = tables["relaxed"].iloc[0]
    assert bool(row["censored"])
    assert row["cc50_relation"] == ">"
    assert row["si"] == pytest.approx(100.0)


def test_each_compound_appears_once():
    frame = _curated(
        _pairable(ASPIRIN, "DOC1", "Vero", 1.0, 100.0)
        + _pairable(ASPIRIN, "DOC2", "HeLa", 10.0, 100.0)
    )
    tables = build(frame)
    assert len(tables["relaxed"]) == 1
    row = tables["relaxed"].iloc[0]
    assert row["n_pairs"] == 2
    # medians of log10([1, 10]) and log10([100, 100])
    assert row["ic50"] == pytest.approx(10**0.5)
    assert row["cc50"] == pytest.approx(100.0)


def test_empty_input_yields_empty_tables():
    frame = _curated([_measurement(measurement_class="unknown", standard_type="Ki")])
    tables = build(frame)
    assert all(tables[level].empty for level in ("strict", "document", "compound", "relaxed"))


def test_si_quality_labels_the_censoring():
    from vacv.build_dataset import si_quality

    assert si_quality("=", "=") == "measured"
    assert si_quality("=", ">") == "lower_bound"
    assert si_quality("=", ">=") == "lower_bound"
    assert si_quality(">", "=") == "upper_bound"
    assert si_quality(">", ">") == "undetermined"
    assert si_quality(">=", ">=") == "undetermined"


def test_both_sides_censored_is_undetermined_not_a_measurement():
    # IC50 > 100 with CC50 > 100 divides out to exactly 1.0 and says nothing.
    frame = _curated(
        [
            _measurement(source_id="ic", value_um=100.0, relation=">"),
            _measurement(
                source_id="cc",
                standard_type="CC50",
                measurement_class="cytotox",
                assay_description="Cytotoxicity against uninfected Vero cells",
                value_um=100.0,
                relation=">",
            ),
        ]
    )
    row = build(frame)["relaxed"].iloc[0]
    assert row["si"] == pytest.approx(1.0)
    assert row["si_quality"] == "undetermined"


def test_si_equals_the_ratio_even_with_several_pairs():
    # Three pairs, because with two the median is the geometric mean and the
    # ratio of geometric means happens to equal the geometric mean of ratios.
    frame = _curated(
        _pairable(ASPIRIN, "DOC1", "Vero", 1.0, 100.0)
        + _pairable(ASPIRIN, "DOC2", "HeLa", 10.0, 20.0)
        + _pairable(ASPIRIN, "DOC3", "BSC-40", 1.0, 1.0)
    )
    row = build(frame)["relaxed"].iloc[0]
    assert row["si"] == pytest.approx(row["cc50"] / row["ic50"])
    assert row["ic50"] == pytest.approx(1.0)
    assert row["cc50"] == pytest.approx(20.0)
    assert row["si"] == pytest.approx(20.0)
    # The median of the three ratios (100, 2, 1) is 2, which is not 20.
    assert row["si_pairwise_median"] == pytest.approx(2.0)


def test_all_pairs_keeps_every_candidate():
    from vacv.build_dataset import all_pairs

    frame = _curated(
        _pairable(ASPIRIN, "DOC1", "Vero", 1.0, 100.0)
        + _pairable(ASPIRIN, "DOC2", "HeLa", 10.0, 50.0)
    )
    pairs = all_pairs(frame)
    assert len(pairs) > 1
    assert set(pairs["pairing_level"]) == {"strict", "document", "compound"}
    assert "si_quality" in pairs.columns
