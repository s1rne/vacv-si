import numpy as np
import pytest

from experiments.chem_zoo import tanimoto_distance, tanimoto_kernel
from vacv.chemistry import BLOCKS, DESCRIPTOR_NAMES, UNSTABLE, block_matrix, features, parse
from vacv.evaluate import SENTINEL, competition_mse, labels_of

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
LONG_CHAIN = "C" * 40
TINY = "C"
SAMPLE = [ASPIRIN, CAFFEINE, LONG_CHAIN, TINY]


def test_unstable_descriptors_excluded():
    assert set(UNSTABLE).isdisjoint(DESCRIPTOR_NAMES)
    assert len(DESCRIPTOR_NAMES) == 215


@pytest.mark.parametrize("name", list(BLOCKS))
def test_blocks_are_finite_and_rectangular(name):
    matrix = block_matrix(SAMPLE, name)
    assert matrix.shape[0] == len(SAMPLE)
    assert np.isfinite(matrix).all()


def test_unstable_descriptors_would_have_blown_up():
    """Исключённые дескрипторы ведут себя неадекватно: Ipc растёт экспоненциально с длиной
    цепи, Kappa3 на малых молекулах уходит в большие отрицательные значения, хотя это
    индекс формы и отрицательным быть не может."""
    from rdkit.Chem import Descriptors, GraphDescriptors

    assert GraphDescriptors.Ipc(parse(LONG_CHAIN)) > 1e8
    assert Descriptors.Kappa3(parse("O")) < -100


def test_features_concatenates_blocks():
    assert features(SAMPLE, ("physchem", "maccs")).shape == (len(SAMPLE), 21 + 167)


def test_parse_rejects_broken_smiles():
    with pytest.raises(ValueError):
        parse("not-a-molecule")


def test_tanimoto_kernel_is_one_on_the_diagonal():
    x = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    kernel = tanimoto_kernel(x, x)
    assert np.allclose(np.diag(kernel), 1.0)
    assert kernel[0, 1] == pytest.approx(2 / (5 + 10 - 2))
    assert tanimoto_distance(x[0], x[1]) == pytest.approx(1 - kernel[0, 1])


def test_tanimoto_kernel_handles_zero_vectors():
    zero = np.zeros((1, 3))
    assert tanimoto_kernel(zero, zero)[0, 0] == 0.0


def test_competition_mse_matches_conditional_mean():
    y = np.array([SENTINEL, 10.0, SENTINEL, 2.0])
    folds = [np.array([0, 1]), np.array([2, 3])]
    probability = np.array([1.0, 0.0, 0.5, 0.5])
    # Средний SI активных берётся по обучающей части фолда: для фолда 0 это строка 3,
    # для фолда 1 — строка 1.
    expected = np.mean([
        (1.0 * SENTINEL - SENTINEL) ** 2,
        (0.0 * SENTINEL + 2.0 - 10.0) ** 2,
        (0.5 * SENTINEL + 0.5 * 10.0 - SENTINEL) ** 2,
        (0.5 * SENTINEL + 0.5 * 10.0 - 2.0) ** 2,
    ])
    assert competition_mse(probability, y, folds) == pytest.approx(expected)


def test_labels_mark_only_the_exact_sentinel():
    y = np.array([SENTINEL, 36066.667, 9999.0])
    assert labels_of(y).tolist() == [1, 0, 0]
