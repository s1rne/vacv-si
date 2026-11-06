import pandas as pd
import pytest

from vacv.validate import assert_ok, errors, validate

GOOD = pd.DataFrame(
    {
        "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"],
        "ic50": [1.0, 2.0],
        "cc50": [100.0, 4.0],
        "si": [100.0, 2.0],
        "doi": ["10.1000/a", None],
        "inchikey14": ["BSYNRYMUTXBX", "RYYVLZVUVIJVG"],
        "pairing_level": ["strict", "compound"],
        "censored": [False, False],
        "high_variance": [False, True],
    }
)


def test_good_frame_passes():
    assert_ok(GOOD)


def test_missing_column_is_an_error():
    findings = validate(GOOD.drop(columns=["si"]))
    assert any("missing required columns" in f.message for f in errors(findings))


def test_unparsable_smiles_is_an_error():
    bad = GOOD.copy()
    bad.loc[0, "smiles"] = "not a molecule ("
    assert any(f.check == "smiles" for f in errors(validate(bad)))


def test_mixture_smiles_is_an_error():
    bad = GOOD.copy()
    bad.loc[0, "smiles"] = "CC(=O)Oc1ccccc1C(=O)O.Cl"
    assert any("mixture" in f.message for f in errors(validate(bad)))


def test_broken_si_is_an_error():
    bad = GOOD.copy()
    bad.loc[0, "si"] = 7.0
    failures = errors(validate(bad))
    assert any(f.check == "si_consistency" for f in failures)


def test_duplicate_compound_is_an_error():
    bad = pd.concat([GOOD, GOOD.iloc[[0]]], ignore_index=True)
    assert any(f.check == "duplicates" for f in errors(validate(bad)))


def test_implausible_units_are_an_error():
    bad = GOOD.copy()
    bad.loc[0, "ic50"] = 1e9
    bad.loc[0, "si"] = bad.loc[0, "cc50"] / bad.loc[0, "ic50"]
    assert any(f.check == "units" for f in errors(validate(bad)))


def test_doi_coverage_is_reported():
    findings = validate(GOOD)
    message = next(f.message for f in findings if f.check == "provenance")
    assert "carry a DOI" in message


def test_assert_ok_raises_on_error():
    bad = GOOD.copy()
    bad.loc[0, "si"] = -1.0
    with pytest.raises(AssertionError):
        assert_ok(bad)
