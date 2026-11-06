import math

import pytest

from vacv.units import canonical_relation, canonical_units, from_pchembl, plausible, to_micromolar


@pytest.mark.parametrize(
    "written,expected",
    [
        ("nM", "nm"),
        ("nmol/L", "nmol/l"),
        ("uM", "um"),
        ("µM", "um"),
        ("μM", "um"),
        ("ug.mL-1", "ug/ml"),
        ("ug ml-1", "ug/ml"),
        ("mg.L-1", "mg/l"),
        ("10'-6M", "um"),
        ("", None),
        (None, None),
    ],
)
def test_canonical_units(written, expected):
    assert canonical_units(written) == expected


def test_molar_conversions():
    assert to_micromolar(1000, "nM")[0] == pytest.approx(1.0)
    assert to_micromolar(1, "uM")[0] == pytest.approx(1.0)
    assert to_micromolar(1, "mM")[0] == pytest.approx(1000.0)
    assert to_micromolar(1e-6, "M")[0] == pytest.approx(1.0)
    assert to_micromolar(5, "nmol/L")[0] == pytest.approx(0.005)


def test_mass_conversion_needs_molecular_weight():
    value, reason = to_micromolar(10, "ug.mL-1")
    assert value is None
    assert reason == "mass_units_without_mw"
    # 10 ug/mL of a 200 g/mol compound is 50 uM.
    assert to_micromolar(10, "ug.mL-1", mol_weight=200.0)[0] == pytest.approx(50.0)
    assert to_micromolar(10, "mg/mL", mol_weight=200.0)[0] == pytest.approx(50000.0)
    assert to_micromolar(10, "ng/mL", mol_weight=200.0)[0] == pytest.approx(0.05)


def test_log_units():
    assert to_micromolar(6.0, "-log M")[0] == pytest.approx(1.0)
    assert to_micromolar(9.0, "log M")[0] == pytest.approx(0.001)


def test_pchembl():
    assert from_pchembl(6.0) == pytest.approx(1.0)
    assert from_pchembl(9.0) == pytest.approx(0.001)
    assert from_pchembl(None) is None
    assert from_pchembl("not a number") is None


def test_refusals_are_explained():
    assert to_micromolar(None, "nM") == (None, "no_value")
    assert to_micromolar("abc", "nM") == (None, "unparsable_value")
    assert to_micromolar(1, None) == (None, "no_units")
    assert to_micromolar(1, "%")[1].startswith("unsupported_units")
    assert to_micromolar(0, "nM") == (None, "non_positive_result")
    assert to_micromolar(float("nan"), "nM") == (None, "non_finite_value")


def test_relations():
    assert canonical_relation(">") == ">"
    assert canonical_relation(">>") == ">"
    assert canonical_relation(None) == "="
    assert canonical_relation("=") == "="
    assert canonical_relation("weird") == "="


def test_plausibility_window():
    assert plausible(1.0)
    assert not plausible(None)
    assert not plausible(1e9)
    assert not plausible(1e-12)
    assert not plausible(math.nan)
