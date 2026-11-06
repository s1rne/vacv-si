import pytest

from vacv.classify import classify, extract_cell_line, extract_strain, mentions_vaccinia


@pytest.mark.parametrize(
    "standard_type,description,expected",
    [
        ("CC50", "cytotoxicity in Vero cells", "cytotox"),
        ("TC50", None, "cytotox"),
        ("IC50", "Inhibition of vaccinia virus replication in Vero cells", "antiviral"),
        ("EC50", "Antiviral activity against Vaccinia virus WR", "antiviral"),
        ("ED50", "Effective dose required for antiviral activity against Vaccinia virus", "antiviral"),
        ("IC50", "Cytotoxicity against uninfected Vero cells", "cytotox"),
        ("IC50", "Minimum cytotoxic concentration in HEL cell cultures", "cytotox"),
        ("IC50", "Inhibition of human carbonic anhydrase II", "unknown"),
        ("Ki", "binding to vaccinia D1 methyltransferase", "unknown"),
        ("IC50", None, "unknown"),
    ],
)
def test_classify(standard_type, description, expected):
    assert classify(standard_type, description=description) == expected


def test_cytotoxicity_measured_as_cpe_is_still_antiviral():
    # Both word groups fire; the specific claim is antiviral unless the text
    # says the cells were uninfected.
    assert (
        classify("EC50", description="Inhibition of vaccinia virus-induced cytopathic effect in Vero cells")
        == "antiviral"
    )
    assert (
        classify("EC50", description="Cytotoxic effect on uninfected Vero cells measured as CPE")
        == "cytotox"
    )


def test_assay_type_is_the_last_resort():
    assert classify("IC50", description="activity in cells", assay_type="T") == "cytotox"
    assert classify("IC50", description="activity in cells", assay_type="B") == "unknown"


def test_mentions_vaccinia_ignores_nan():
    assert mentions_vaccinia("antiviral activity against Vaccinia virus")
    assert mentions_vaccinia(None, float("nan"), "VACV WR")
    assert not mentions_vaccinia(None, float("nan"))
    assert not mentions_vaccinia("activity against herpes simplex virus 1")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Vaccinia virus strain WR", "WR"),
        ("vaccinia virus (Western Reserve)", "WR"),
        ("VACV Copenhagen", "Copenhagen"),
        ("vaccinia IHD-J", "IHD-J"),
        ("modified vaccinia Ankara", "Ankara"),
        ("vaccinia virus", None),
    ],
)
def test_extract_strain(text, expected):
    assert extract_strain(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("cytotoxicity in Vero cells", "Vero"),
        ("plaque reduction in BSC-40 monolayers", "BSC-40"),
        ("BSC-1 cells", "BSC-1"),
        ("human foreskin fibroblasts", "HFF"),
        ("HeLa cell viability", "HeLa"),
        ("in mice", None),
    ],
)
def test_extract_cell_line(text, expected):
    assert extract_cell_line(text) == expected
