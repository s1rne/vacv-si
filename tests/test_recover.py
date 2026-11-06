import numpy as np
import pandas as pd
import pytest

from vacv import coverage, recover
from vacv.match_competition import open_source_rows


def measurement(**overrides):
    row = {
        "smiles": "CCO",
        "smiles_std": "CCO",
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "inchikey14": "LFQSCWFLJHTTHZ",
        "standard_type": "EC50",
        "measurement_class": "antiviral",
        "value_um": 1.0,
        "relation": "=",
        "censored": False,
        "assay_description": "Inhibition of vaccinia virus replication in Vero cells",
        "assay_format": "cell-based format",
        "organism": "Vaccinia virus",
        "strain": "WR",
        "cell_line": "Vero",
        "document_id": "CHEMBL1",
        "doi": "10.1000/one",
        "year": 2000,
        "source_db": "ChEMBL",
        "source_id": "CHEMBL_ACT_1",
        "raw_units": "nM",
        "raw_value": 1000.0,
    }
    row.update(overrides)
    return row


def split_row(**overrides):
    row = {
        "split": "test",
        "index": 0,
        "Title": "1",
        "SMILES": "CCO",
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "inchikey14": "LFQSCWFLJHTTHZ",
        "Culture": "Vero",
        "strain": "Vaccinia virus WR",
    }
    row.update(overrides)
    return row


def test_the_printed_number_is_used_for_mass_concentrations():
    frame = pd.DataFrame(
        [
            measurement(value_um=404.0, raw_value=100.0, raw_units="ug.mL-1"),
            measurement(value_um=2.0, raw_value=2000.0, raw_units="nM"),
        ]
    )
    printed = recover.printed_values(frame)
    # 100 ug/mL went into their column as 100, not as the 404 uM it is.
    assert printed.iloc[0] == pytest.approx(100.0)
    assert printed.iloc[1] == pytest.approx(2.0)


def test_prefers_the_named_cell_line_then_the_censored_record():
    records = [
        measurement(value_um=5.0, cell_line="HeLa", source_id="a"),
        measurement(value_um=7.0, cell_line="Vero", censored=False, source_id="b"),
        measurement(value_um=100.0, cell_line="Vero", censored=True, relation=">", source_id="c"),
    ]
    assert recover._best(records, "Vero")["value_um"] == 100.0
    assert recover._best(records, "HeLa")["value_um"] == 5.0


def test_pairs_inside_one_publication_and_falls_back_to_the_compound():
    antiviral = [measurement(value_um=2.0, document_id="CHEMBL1")]
    cytotox = [
        measurement(measurement_class="cytotox", standard_type="CC50", value_um=50.0, document_id="CHEMBL1"),
        measurement(measurement_class="cytotox", standard_type="CC50", value_um=900.0, document_id="CHEMBL2"),
    ]
    found = recover._pair(antiviral, cytotox, "Vero")
    assert found["rule"] == "document, same cell line"
    assert found["cytotox"]["value_um"] == 50.0

    elsewhere = [measurement(measurement_class="cytotox", value_um=900.0, document_id="CHEMBL2")]
    found = recover._pair(antiviral, elsewhere, "Vero")
    assert found["rule"] == "compound"
    assert found["document_id"] is None

    assert recover._pair(antiviral, [], "Vero") is None


def test_recovers_the_pair_and_the_quality_of_its_ratio():
    curated = pd.DataFrame(
        [
            measurement(value_um=2.0),
            measurement(
                measurement_class="cytotox",
                standard_type="CC50",
                value_um=100.0,
                relation=">",
                censored=True,
                source_id="CHEMBL_ACT_2",
            ),
        ]
    )
    table = recover.recover(pd.DataFrame([split_row()]), recover.Records(curated), pd.DataFrame())
    assert table.loc[0, "si"] == pytest.approx(50.0)
    assert table.loc[0, "match_tier"] == "inchikey"
    assert table.loc[0, "rule"] == "document, same cell line"
    # CC50 is a lower bound, so the ratio is one too.
    assert table.loc[0, "si_quality"] == "lower_bound"
    assert not table.loc[0, "sentinel_evidence"]


def test_cytotoxicity_without_an_antiviral_record_is_flagged_not_valued():
    curated = pd.DataFrame(
        [measurement(measurement_class="cytotox", standard_type="CC50", value_um=45.0)]
    )
    table = recover.recover(pd.DataFrame([split_row()]), recover.Records(curated), pd.DataFrame())
    assert pd.isna(table.loc[0, "si"])
    assert bool(table.loc[0, "sentinel_evidence"])


def test_reported_selectivity_index_is_used_only_as_a_fallback():
    reported = measurement(
        measurement_class="selectivity",
        standard_type="Ratio CC50/EC50",
        value_um=146.0,
        raw_units=None,
        source_id="CHEMBL_ACT_3",
    )
    table = recover.recover(
        pd.DataFrame([split_row()]), recover.Records(pd.DataFrame([reported])), pd.DataFrame()
    )
    assert table.loc[0, "si"] == pytest.approx(146.0)
    assert table.loc[0, "rule"] == "reported ratio"
    assert table.loc[0, "si_quality"] == "reported_ratio"

    with_pair = pd.DataFrame(
        [
            reported,
            measurement(value_um=2.0),
            measurement(
                measurement_class="cytotox", standard_type="CC50", value_um=50.0, source_id="CHEMBL_ACT_4"
            ),
        ]
    )
    table = recover.recover(pd.DataFrame([split_row()]), recover.Records(with_pair), pd.DataFrame())
    assert table.loc[0, "rule"] == "document, same cell line"
    assert table.loc[0, "si"] == pytest.approx(25.0)


def test_a_selectivity_index_for_another_virus_is_not_used():
    reported = measurement(
        measurement_class="selectivity",
        standard_type="Selectivity index",
        value_um=146.0,
        organism="Human immunodeficiency virus 1",
        assay_description="Selectivity index against HIV-1 in MT-4 cells",
        strain=None,
    )
    table = recover.recover(
        pd.DataFrame([split_row()]), recover.Records(pd.DataFrame([reported])), pd.DataFrame()
    )
    assert pd.isna(table.loc[0, "si"])


def test_the_competition_rows_cannot_reach_the_recovered_values():
    """The organisers' own numbers must never come back as "recovered"."""
    curated = pd.DataFrame(
        [
            measurement(value_um=2.0, source_db="Kaggle competition", source_id="KAGGLE_1"),
            measurement(
                measurement_class="cytotox",
                standard_type="CC50",
                value_um=4.0,
                source_db="Kaggle competition",
                source_id="KAGGLE_2",
            ),
            measurement(value_um=10.0, source_id="CHEMBL_ACT_9"),
            measurement(
                measurement_class="cytotox",
                standard_type="CC50",
                value_um=1000.0,
                source_id="CHEMBL_ACT_10",
            ),
        ]
    )
    records = recover.Records(open_source_rows(curated))
    table = recover.recover(pd.DataFrame([split_row()]), records, pd.DataFrame())
    # 1000/10 from the open rows, never 4/2 from the competition's own.
    assert table.loc[0, "si"] == pytest.approx(100.0)


def test_labels_are_joined_positionally_and_refuse_to_misalign():
    lookup = pd.DataFrame(
        [
            {"split": "train", "Title": "not in the file", "si": 1.0, "si_quality": "measured",
             "ic50": 1.0, "cc50": 1.0, "match_tier": "inchikey", "rule": "document",
             "sentinel_evidence": False, "n_antiviral_records": 1, "n_cytotox_records": 1},
        ]
    )
    with pytest.raises(ValueError):
        coverage.join_labels(lookup, "train")


def test_confidence_falls_back_to_the_overall_rate_for_thin_buckets():
    labelled = pd.DataFrame(
        {
            "si": [1.0] * 6,
            "match_tier": ["inchikey"] * 6,
            "rule": ["document"] * 5 + ["compound"],
            "is_sentinel": [False] * 6,
            "log10_error": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        }
    )
    lookup = pd.DataFrame(
        {
            "match_tier": ["inchikey", "inchikey", None],
            "rule": ["document", "compound", None],
        }
    )
    values = coverage.confidence(lookup, labelled)
    assert values.iloc[0] == pytest.approx(0.8)
    # One row is too few to estimate from, so the bucket inherits the overall rate.
    assert values.iloc[1] == pytest.approx(5 / 6)
    assert np.isnan(values.iloc[2])
