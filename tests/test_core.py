import numpy as np

from vacv.features import descriptors, maccs, morgan_counts, parse, scaffold
from vacv.splits import fold_indices, scaffold_folds
from vacv.target import censored_mse, from_log, is_censored, to_log

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_features_shapes():
    mol = parse(ASPIRIN)
    assert mol is not None
    assert descriptors(mol).ndim == 1 and np.isfinite(descriptors(mol)).all()
    assert morgan_counts(mol).shape == (2048,)
    assert maccs(mol).shape == (167,)
    assert scaffold(mol) == "c1ccccc1"


def test_parse_rejects_garbage():
    assert parse("не молекула") is None
    assert parse(None) is None


def test_log_roundtrip():
    values = np.array([0.5, 1.0, 250.0, 10_000.0])
    assert np.allclose(from_log(to_log(values)), values)


def test_censoring_flags_sentinel():
    assert is_censored(np.array([10_000.0, 9_999.0])).tolist() == [True, False]


def test_censored_error_ignores_overprediction_for_inactive():
    target = np.array([4.0, 4.0, 2.0])
    prediction = np.array([5.0, 3.0, 3.0])
    censored = np.array([True, True, False])
    assert censored_mse(prediction, target, censored).tolist() == [0.0, 1.0, 1.0]


def test_scaffold_folds_keep_groups_together():
    scaffolds = ["a", "a", "b", "b", "c", "d", "e", "f"]
    folds = scaffold_folds(scaffolds, n_splits=3)
    assert sum(len(f) for f in folds) == len(scaffolds)
    position = {index: i for i, fold in enumerate(folds) for index in fold}
    assert position[0] == position[1] and position[2] == position[3]


def test_fold_indices_split_is_disjoint():
    folds = scaffold_folds([str(i) for i in range(10)], n_splits=5)
    train, validation = fold_indices(folds, 0)
    assert set(train).isdisjoint(validation)
    assert len(train) + len(validation) == 10
