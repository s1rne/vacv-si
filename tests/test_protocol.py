import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from vacv.neural.augment import expand, randomized
from vacv.neural.ensemble import blend
from vacv.neural.external import skeleton
from vacv.protocol import SENTINEL, Result, conditional_mean, folds_of, inactive_labels, score, task

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
TAMIFLU = "CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@H](NC(C)=O)[C@@H](N)C1"


def frame(smiles, values):
    return pd.DataFrame({"SMILES": smiles, "SI VACV": values})


def test_sentinel_label_is_exact_not_a_threshold():
    """SI = 36066 — настоящее измерение, порог «не меньше 10000» относил бы его к неактивным."""
    labels = inactive_labels(np.array([1.0, SENTINEL, 36066.667, 9999.0]))
    assert list(labels) == [0, 1, 0, 0]


def test_conditional_mean_interpolates_between_sentinel_and_active_estimate():
    assert conditional_mean(np.array([0.0, 1.0, 0.5]), 100.0).tolist() == [100.0, 10000.0, 5050.0]


def test_conditional_mean_accepts_per_row_active_estimate():
    got = conditional_mean(np.array([0.5, 0.5]), np.array([0.0, 100.0]))
    assert got.tolist() == [5000.0, 5050.0]


def test_conditional_mean_clips_probability():
    assert conditional_mean(np.array([-0.3, 1.4]), 0.0).tolist() == [0.0, 10000.0]


def test_score_returns_auc_and_competition_mse():
    y = np.array([1.0, 2.0, SENTINEL, SENTINEL])
    labels = inactive_labels(y)
    auc, mse = score(labels, y, np.array([0.0, 0.0, 1.0, 1.0]), 1.5)
    assert auc == 1.0
    assert mse == pytest.approx(0.125)


def test_folds_cover_every_row_exactly_once():
    data = frame([ASPIRIN, CAFFEINE, IBUPROFEN, TAMIFLU] * 5, [1.0] * 20)
    for kind in ("random", "scaffold"):
        folds = folds_of(data, kind)
        assert len(folds) == 5
        assert sorted(np.concatenate(folds)) == list(range(20))


def test_scaffold_folds_keep_one_scaffold_in_one_fold():
    smiles = [ASPIRIN] * 4 + [CAFFEINE] * 4 + [IBUPROFEN] * 4 + [TAMIFLU] * 4
    folds = folds_of(frame(smiles, [1.0] * 16), "scaffold")
    membership = {index: fold for fold, members in enumerate(folds) for index in members}
    for start in (0, 4, 8, 12):
        assert len({membership[start + offset] for offset in range(4)}) == 1


def test_folds_are_reproducible():
    data = frame([ASPIRIN, CAFFEINE, IBUPROFEN, TAMIFLU] * 5, [1.0] * 20)
    for kind in ("random", "scaffold"):
        first, second = folds_of(data, kind), folds_of(data, kind)
        assert [f.tolist() for f in first] == [f.tolist() for f in second]


def test_task_splits_hold_out_each_fold_once():
    data = frame([ASPIRIN, CAFFEINE, IBUPROFEN, TAMIFLU] * 5, [1.0] * 19 + [SENTINEL])
    job = task(data)
    seen = []
    for train_index, validation_index in job.splits("random"):
        assert not set(train_index) & set(validation_index)
        assert len(train_index) + len(validation_index) == 20
        seen += list(validation_index)
    assert sorted(seen) == list(range(20))


def test_randomized_smiles_are_distinct_strings_for_one_molecule():
    variants = randomized(IBUPROFEN, 8)
    assert len(set(variants)) > 1
    canonical = {Chem.MolToSmiles(Chem.MolFromSmiles(text)) for text in variants}
    assert canonical == {Chem.MolToSmiles(Chem.MolFromSmiles(IBUPROFEN))}


def test_randomized_smiles_returns_requested_count():
    assert len(randomized("C", 5)) == 5


def test_expand_repeats_labels_alongside_variants():
    texts, labels = expand([ASPIRIN, CAFFEINE], np.array([0.0, 1.0]), 3)
    assert len(texts) == 6
    assert labels.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_skeleton_ignores_stereochemistry():
    flat = "CC(N)C(=O)O"
    chiral = "C[C@H](N)C(=O)O"
    assert skeleton(flat) == skeleton(chiral)
    assert skeleton(ASPIRIN) != skeleton(CAFFEINE)


def test_skeleton_of_unparsable_smiles_is_empty():
    assert skeleton("not a molecule") == ""


def test_blend_weight_favours_the_model_with_lower_mse():
    y = np.array([1.0, 2.0, SENTINEL, SENTINEL] * 2 + [3.0, SENTINEL])
    job = task(frame([ASPIRIN, CAFFEINE, IBUPROFEN, TAMIFLU] * 2 + [ASPIRIN, CAFFEINE], y))
    good = (y == SENTINEL).astype(float)
    poor = np.where(y == SENTINEL, 0.6, 0.4)
    base = Result("base", "random", *score(job.labels, y, good, job.active_mean), 0.0, good)
    candidate = Result("candidate", "random", *score(job.labels, y, poor, job.active_mean), 0.0, poor)
    mixed = blend(job, base, candidate)
    assert mixed.base_mse < mixed.inverse_mse
    assert mixed.best_weight == 0.0
    assert mixed.best_mse == pytest.approx(mixed.base_mse)
