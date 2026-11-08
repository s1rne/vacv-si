import numpy as np
import pandas as pd

from experiments.v2_analog import fingerprint
from experiments.v10_spec_validation import calibrate, evaluate_fold, fingerprints, variants


def test_scalar_solution_zero_fallback_nonnegative_and_no_arbitrary_upper_bound():
    p = np.array([1., 2., 3.])
    assert calibrate(20 * p, p) == 20.
    assert calibrate(p, np.zeros(3)) == 1.
    assert calibrate(-p, p) == 0.
    alpha = calibrate(np.array([2., 7., 5.]), p)
    loss = lambda a: np.sum((a * p - np.array([2., 7., 5.])) ** 2)
    assert loss(alpha) < loss(alpha - .01)
    assert loss(alpha) < loss(alpha + .01)


def test_outer_holdout_labels_never_affect_calibration_or_predictions():
    y = np.array([1., 4., 9., 16., 25., 36.])
    tr, va = np.arange(4), np.array([4, 5])
    inner = [{'train': [0, 1], 'valid': [2, 3]}, {'train': [2, 3], 'valid': [0, 1]}]

    def predict(a, b, labels):
        assert np.isnan(labels[np.setdiff1d(np.arange(len(y)), a)]).all()
        return np.full(len(b), labels[a].mean())

    before = evaluate_fold(y, tr, va, inner, predict)
    changed = y.copy()
    changed[va] = [1e9, 1e12]
    after = evaluate_fold(changed, tr, va, inner, predict)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(before[2], [12.5, 12.5, 2.5, 2.5])


def test_prediction_cap_does_not_modify_targets_or_original_predictions():
    y = np.array([1., 40000.])
    original = y.copy()
    p = np.array([100., 12000.])
    result = variants(p, 2.)
    np.testing.assert_array_equal(y, original)
    np.testing.assert_array_equal(p, [100., 12000.])
    np.testing.assert_array_equal(result[:, 2], [100., 5000.])
    np.testing.assert_array_equal(result[:, 3], [200., 5000.])
    assert np.mean((result[:, 2] - y)**2) > 6e8


def test_chirality_binary_fingerprint_distinguishes_toy_stereoisomers():
    smiles = ['N[C@@H](C)C(=O)O', 'N[C@H](C)C(=O)O']
    original, chiral = fingerprints(smiles), fingerprints(smiles, True)
    np.testing.assert_array_equal(original[0], original[1])
    assert not np.array_equal(chiral[0], chiral[1])
    assert set(np.unique(chiral)) <= {0., 1.}
    np.testing.assert_array_equal(original, fingerprint(pd.DataFrame({'SMILES': smiles})))
