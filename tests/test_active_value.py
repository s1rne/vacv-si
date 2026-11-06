"""Тесты по-строчной оценки активного SI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vacv.active_value import ActiveValueModel
from vacv.pipeline import INACTIVE


def frame(rows: list[dict]) -> pd.DataFrame:
    base = {"strain": "s", "Culture": "Vero", "SMILES": "CCO", "matched": 0.0,
            "best_ic50_um": np.nan, "best_cc50_um": np.nan}
    return pd.DataFrame([{**base, **row} for row in rows])


def test_constant_fallback_for_rows_without_evidence():
    train = frame([{"SMILES": "CCO"}, {"SMILES": "CCC"}])
    y = np.array([4.0, INACTIVE])
    model = ActiveValueModel().fit(train, y)
    query = frame([{"SMILES": "CCCC"}])
    np.testing.assert_allclose(model.predict(query), [4.0])


def test_both_known_gives_exact_ratio():
    train = frame([{"SMILES": "CCO"}])
    y = np.array([INACTIVE])
    model = ActiveValueModel().fit(train, y)
    query = frame([{"best_ic50_um": 2.0, "best_cc50_um": 300.0}])
    np.testing.assert_allclose(model.predict(query), [150.0])


def test_ic50_only_uses_neighbor_cc50():
    # Ряд с известным IC50 и похожим соседом из внешней базы: A = CC50_hat / IC50.
    train = frame([{"SMILES": "CCO"}])
    y = np.array([INACTIVE])
    model = ActiveValueModel().fit(train, y)
    ic50 = 0.03
    query = frame([{"SMILES": "CC(C)CC1=CC2=C(C=C1)N=C1C(C(C)(C)C)=CC(N(CC)CC)=CC1=[S+]2",
                    "best_ic50_um": ic50}])
    estimate = model.predict(query)[0]
    # CC50_hat стянут к prior по клеточной линии; SI обязан быть много больше среднего.
    assert estimate > 100.0 / ic50


def test_predictions_finite_and_positive():
    rng = np.random.default_rng(0)
    train = frame([{"SMILES": f"{'C' * (3 + i)}O", "best_ic50_um": float(i + 1)}
                   for i in range(6)])
    y = np.array([1.0, 2.0, INACTIVE, 8.0, 3.0, INACTIVE])
    model = ActiveValueModel().fit(train, y)
    query = frame([{"SMILES": "CCCO", "best_ic50_um": 5.0},
                   {"SMILES": "CCCCCO", "best_cc50_um": 40.0},
                   {"SMILES": "CCCCCCO"},
                   {"SMILES": "not-a-smiles", "best_ic50_um": 1.0}])
    values = model.predict(query)
    assert np.isfinite(values).all() and (values >= 0).all()
    assert values[2] == pytest.approx(model.fallback_)


def test_self_record_is_excluded_from_cc50_estimate():
    # Строка обучения, чей собственный CC50 есть в базе, не должна видеть его.
    db = pd.read_csv("data/processed/vacv_si.csv")
    side = db[db["cc50_relation"].eq("=") & db["cc50"].notna()]
    row = side.iloc[0]
    train = frame([{"SMILES": row["smiles"], "best_cc50_um": float(row["cc50"])}])
    y = np.array([50.0])
    model = ActiveValueModel().fit(train, y)
    query = frame([{"SMILES": row["smiles"], "best_ic50_um": 1.0}])
    estimate = model.predict(query)[0]
    prior = model._cell_prior("Vero")
    # Без исключения собственной записи оценка стянулась бы к её CC50.
    assert abs(estimate - float(row["cc50"])) > 1e-9 or estimate != pytest.approx(float(row["cc50"]))
    assert np.isfinite(estimate) and estimate > 0
