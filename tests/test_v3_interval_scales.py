import numpy as np
import pytest

from experiments.v3_interval_scales import transform, inverse


@pytest.mark.parametrize("scale", ["raw", "sqrt"])
def test_scale_preserves_targets_and_infinite_bound(scale):
    targets = np.array([0., 1., 5000., 36066.667, 157026., np.inf])
    np.testing.assert_allclose(inverse(transform(targets, scale), scale), targets, rtol=1e-12)
    assert np.isposinf(transform(targets, scale)[-1])
    assert inverse(np.array([-2., 0., 2.]), scale)[0] == 0


@pytest.mark.parametrize("scale", ["raw", "sqrt"])
def test_invalid_targets_are_not_silently_fixed(scale):
    for targets in ([np.nan], [-1.]):
        with pytest.raises(ValueError):
            transform(targets, scale)
