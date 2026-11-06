import numpy as np
import pandas as pd
import pytest

from vacv.curate import VARIANCE_FLAG_LOG10, aggregate, curate, standardize
from vacv.schema import conform


def test_salt_is_stripped_to_the_parent():
    free = standardize("CN1CCC[C@H]1c1cccnc1")
    salt = standardize("Cl.CN1CCC[C@H]1c1cccnc1")
    assert free.inchikey is not None
    assert salt.inchikey == free.inchikey


def test_charges_are_neutralised():
    acid = standardize("CC(=O)Oc1ccccc1C(=O)O")
    anion = standardize("CC(=O)Oc1ccccc1C(=O)[O-]")
    assert anion.inchikey == acid.inchikey


def test_rejections():
    assert standardize(None).reject == "empty_smiles"
    assert standardize("not a smiles").reject == "unparsable_smiles"
    assert standardize("O").reject == "inorganic"
    assert standardize("[Fe]").reject in {"inorganic", "disallowed_elements:Fe"}
    assert standardize("CC").reject == "too_few_heavy_atoms"


def test_mixture_of_two_organics_keeps_the_largest_fragment():
    # FragmentParent plus LargestFragmentChooser is the documented behaviour:
    # a co-crystal or a salt of two organics resolves to the bigger half
    # rather than being thrown away.
    result = standardize("CCO.c1ccc2c(c1)ccc1ccccc12")
    assert result.reject is None
    assert result.smiles == standardize("c1ccc2c(c1)ccc1ccccc12").smiles


def _frame(rows):
    return conform(pd.DataFrame(rows))


def test_curate_recovers_mass_units_with_computed_mw():
    frame = _frame(
        [
            {
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",  # MW 180.16
                "standard_type": "IC50",
                "measurement_class": "antiviral",
                "value_um": None,
                "relation": "=",
                "raw_value": 18.016,
                "raw_units": "ug.mL-1",
                "assay_description": "antiviral activity against vaccinia virus",
                "source_db": "test",
                "source_id": "t1",
            }
        ]
    )
    kept, reasons = curate(frame, canonical_tautomer=False)
    assert len(kept) == 1
    assert kept.loc[0, "value_um"] == pytest.approx(100.0, rel=1e-3)
    assert not reasons


def test_curate_counts_every_rejection():
    frame = _frame(
        [
            {"smiles": "garbage(", "standard_type": "IC50", "measurement_class": "antiviral",
             "value_um": 1.0, "relation": "=", "source_db": "t", "source_id": "a"},
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "standard_type": "IC50",
             "measurement_class": "antiviral", "value_um": None, "relation": "=",
             "source_db": "t", "source_id": "b"},
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "standard_type": "IC50",
             "measurement_class": "antiviral", "value_um": 1e12, "relation": "=",
             "source_db": "t", "source_id": "c"},
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "standard_type": "Ki",
             "measurement_class": "unknown", "value_um": 1.0, "relation": "=",
             "source_db": "t", "source_id": "d"},
        ]
    )
    kept, reasons = curate(frame, canonical_tautomer=False)
    assert len(kept) == 0
    assert reasons["unparsable_smiles"] == 1
    assert reasons["no_micromolar_value"] == 1
    assert reasons["value_outside_plausible_range"] == 1
    assert reasons["unclassifiable_endpoint"] == 1


def test_aggregate_takes_the_median_in_log_space():
    rows = []
    for index, value in enumerate([0.1, 1.0, 10.0, 1000.0]):
        rows.append(
            {
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "standard_type": "IC50",
                "measurement_class": "antiviral",
                "value_um": value,
                "relation": "=",
                "source_db": "test",
                "source_id": f"s{index}",
                "document_id": "D1",
            }
        )
    kept, _ = curate(_frame(rows), canonical_tautomer=False)
    collapsed = aggregate(kept, ["inchikey14"])
    assert len(collapsed) == 1
    # median of log10 [-1, 0, 1, 3] is 0.5 -> 10 ** 0.5
    assert collapsed.loc[0, "value_um"] == pytest.approx(10**0.5)
    assert collapsed.loc[0, "n_measurements"] == 4
    assert collapsed.loc[0, "log10_spread"] == pytest.approx(4.0)
    assert bool(collapsed.loc[0, "high_variance"])


def test_aggregate_does_not_flag_tight_replicates():
    rows = [
        {
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "standard_type": "IC50",
            "measurement_class": "antiviral",
            "value_um": value,
            "relation": "=",
            "source_db": "test",
            "source_id": f"s{index}",
        }
        for index, value in enumerate([1.0, 1.5, 2.0])
    ]
    kept, _ = curate(_frame(rows), canonical_tautomer=False)
    collapsed = aggregate(kept, ["inchikey14"])
    assert collapsed.loc[0, "log10_spread"] < VARIANCE_FLAG_LOG10
    assert not bool(collapsed.loc[0, "high_variance"])


def test_censored_flag_follows_the_relation():
    frame = _frame(
        [
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "standard_type": "CC50",
             "measurement_class": "cytotox", "value_um": 100.0, "relation": ">",
             "source_db": "t", "source_id": "a"},
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "standard_type": "IC50",
             "measurement_class": "antiviral", "value_um": 1.0, "relation": "=",
             "source_db": "t", "source_id": "b",
             "assay_description": "antiviral against vaccinia"},
        ]
    )
    kept, _ = curate(frame, canonical_tautomer=False)
    assert kept.set_index("source_id").loc["a", "censored"]
    assert not kept.set_index("source_id").loc["b", "censored"]


def test_inchikey14_ignores_stereochemistry():
    a = standardize("C[C@H](N)C(=O)Nc1ccccc1")
    b = standardize("C[C@@H](N)C(=O)Nc1ccccc1")
    assert a.inchikey != b.inchikey
    assert a.inchikey[:14] == b.inchikey[:14]


def test_log_space_median_is_not_the_arithmetic_mean():
    values = np.array([0.01, 100.0])
    assert 10 ** np.median(np.log10(values)) == pytest.approx(1.0)
    assert values.mean() == pytest.approx(50.005)
