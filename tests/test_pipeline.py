import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier

from vacv.dataset import submission
from vacv.inference import predict_ensemble
from vacv.literature import apply
from vacv.pipeline import SelectivityModel, crossfit_group_rate, group_rate, structure_matrix
from vacv.kernel import KernelSelectivityModel


def frame(n=12):
    return pd.DataFrame({
        "SMILES": ["CCO", "CCN", "CCC"] * (n // 3),
        "strain": ["a"] * n, "Culture": ["cells"] * n,
        "Title": [str(i) for i in range(n)], "literature_si": [np.nan] * n,
    })


def test_crossfit_row_does_not_see_its_own_target_or_prior():
    x = frame()
    y = np.array([0, 1] * 6)
    original = crossfit_group_rate(x, y, seed=17)
    for i in range(len(y)):
        changed = y.copy()
        changed[i] = 1 - changed[i]
        assert crossfit_group_rate(x, changed, seed=17)[i] == original[i]


def test_unseen_group_uses_only_training_prior():
    x = frame()
    test = x.iloc[:1].assign(strain="unseen")
    assert group_rate(x, np.array([0, 1] * 6), test)[0] == pytest.approx(0.5)


def test_single_class_and_large_real_si_are_finite():
    x = frame()
    inactive = SelectivityModel().fit(x, np.full(len(x), 10000.0))
    assert np.all(inactive.predict(x) == 10000)
    # A real measured SI above the sentinel is active, not class 1.
    y = np.array([36066.0] + [1.0] * 11)
    active = SelectivityModel().fit(x, y)
    assert np.all(active.probability(x) == 0)
    assert np.allclose(active.predict(x), y.mean())


def test_invalid_smiles_preserves_output_rows():
    x = frame(3)
    x.loc[1, "SMILES"] = "invalid(smiles"
    matrix = structure_matrix(x)
    assert matrix.shape == (3, 2215)
    assert np.isfinite(matrix).all()
    assert np.count_nonzero(matrix[1]) == 0


def test_shared_ensemble_matches_manual_submission(monkeypatch):
    monkeypatch.setattr(SelectivityModel, "_forest", lambda self: ExtraTreesClassifier(
        n_estimators=12, min_samples_leaf=2, random_state=self.seed))
    train = frame()
    test = train.iloc[:3].copy()
    test.loc[test.index[0], "literature_si"] = 42.0
    y = np.array([1, 10000] * 6, dtype=float)
    weights = {"structure": 0.0, "context": 1.0, "grouped": 0.0}
    expected = np.mean([SelectivityModel(seed=i, weights=weights).fit(train, y).predict(test)
                        for i in range(2)], axis=0)
    probability, raw, final = predict_ensemble(train, y, test, seeds=2, profile="context")
    assert np.allclose(raw, expected)
    assert np.allclose(final, apply(expected, test))
    assert final[0] == 42
    assert ((probability >= 0) & (probability <= 1)).all()


@pytest.mark.parametrize("prediction", [[1], [np.nan, 2], [-1, 2], [np.inf, 2]])
def test_submission_rejects_invalid_predictions(prediction):
    with pytest.raises(ValueError):
        submission(np.array([1, 2]), np.array(prediction))


def test_submission_keeps_valid_large_si_and_index_order():
    result = submission(np.array([9, 3]), np.array([36066, 4]))
    assert result["index"].tolist() == [9, 3]
    assert result["SI VACV"].tolist() == [36066, 4]


def test_submission_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        submission(np.array([1, 1]), np.array([1, 2]))


def test_kernel_probabilities_for_unseen_context_and_invalid_molecule():
    x = frame()
    model = KernelSelectivityModel().fit(x, np.array([1, 10000] * 6))
    test = x.iloc[:3].assign(strain="unseen").copy()
    test.loc[test.index[0], "SMILES"] = "bad(smiles"
    probability = model.probability(test)
    assert probability.shape == (3,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_cv_auc_uses_exact_sentinel_and_classifier_probability(monkeypatch):
    from scripts import cv

    x = frame(3)
    y = np.array([10000.0, 36066.0, 1.0])
    def predict(train, target, valid, seeds, profile):
        probability = np.array([0.9, 0.1, 0.2])[valid.index]
        # Deliberately different ranking after the external-data override.
        prediction = np.array([4.0, 36066.0, 1.0])[valid.index]
        return probability, probability * 10000, prediction
    monkeypatch.setattr(cv, "predict_ensemble", predict)
    metrics, oof = cv.evaluate(x, y, [np.array([0]), np.array([1, 2])], seeds=1)
    assert metrics["auc"] == 1.0
    assert metrics["mse"] == pytest.approx((10000 - 4) ** 2 / 3)
    assert oof["index"].tolist() == [0, 1, 2]


def test_loader_drops_forbidden_columns(tmp_path):
    from vacv.dataset import FORBIDDEN, load

    x = frame(3).assign(**{"SI VACV": [1, 10000, 36066]})
    for column in FORBIDDEN:
        x[column] = [999, 0, np.nan]
    x.to_csv(tmp_path / "train.csv", index=False)
    x.to_csv(tmp_path / "test.csv", index=False)
    data = load(tmp_path)
    assert not set(FORBIDDEN) & set(data.train.columns)
    assert not set(FORBIDDEN) & set(data.test.columns)
