"""Тесты хард-оверрайдов из таблиц статей."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vacv.article_lookup import apply_article_overrides, article_frame, attach_article


def test_article_table_covers_series():
    art = article_frame()
    assert len(art) >= 26
    assert art["smiles"].is_unique
    # 24a11: EC50 2.4, MCC>100 → SI = 100/2.4
    row = art[art["Title"] == "24a11"].iloc[0]
    assert not row["sentinel"]
    assert row["si"] == pytest.approx(100.0 / 2.4, rel=1e-3)
    # 24b08: EC50 >100 → сентинел, несмотря на измеренный CC50=20
    row = art[art["Title"] == "24b08"].iloc[0]
    assert row["sentinel"] and row["cc50_um"] == 20.0


def test_paper_overrides_present():
    """Пять статей, сматченных через ChEMBL/лейблы: сентинелы и ratio-ряды."""
    art = article_frame()
    # Title '9a' встречается трижды (train камфора-амид SI=13.31 и два test
    # имидазопиридина-сентинела) — проверяем по лейблам статей
    for title, pmc in [("18a", "PMC9865139"), ("16a", "PMC9865139"),
                       ("9a", "PMC11331333"), ("4g", "PMC6194496"), ("38", "PMC9533879")]:
        assert (art["Title"] == title).any(), title
        assert (art.loc[art["Title"] == title, "pmc"] == pmc).any(), title
    row = art[art["Title"] == "18a"].iloc[0]
    assert not row["sentinel"]
    assert row["si"] == pytest.approx(546.43 / 0.03, rel=1e-3)
    assert art.loc[art["Title"] == "16a", "sentinel"].all()


def test_overrides_replace_prediction():
    art = article_frame().set_index("smiles")
    sentinel_smiles = art[art["sentinel"]].index[0]
    si_row = art[art["si"].notna()].iloc[0]
    frame = pd.DataFrame({"SMILES": [sentinel_smiles, si_row.name, "CCCCO"]})
    prediction = np.array([5000.0, 3000.0, 7000.0])
    out = apply_article_overrides(frame, prediction)
    assert out[0] == 10000.0
    assert out[1] == pytest.approx(float(si_row["si"]))
    assert out[2] == 7000.0  # без переопределения — без изменений
    # исходный массив не изменён
    assert prediction[0] == 5000.0


def test_attach_article_flags():
    art = article_frame()
    frame = pd.DataFrame({"SMILES": [art.iloc[0]["smiles"], "CCCCO"]})
    out = attach_article(frame)
    assert out["article_sentinel"].tolist() == [bool(art.iloc[0]["sentinel"]), False]
    assert np.isnan(out["article_si"].iloc[1])
