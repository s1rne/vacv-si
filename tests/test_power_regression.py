import numpy as np
import pandas as pd
import pytest
from vacv.power_regression import PowerRegression, transform, inverse


@pytest.mark.parametrize('power', [0, .25, .5, 1])
def test_transform_retains_zero_and_extreme_measured_target(power):
    y = np.array([0., 1., 10000., 36066.667])
    np.testing.assert_allclose(inverse(transform(y, power), power), y, rtol=1e-12, atol=1e-12)


def test_predictions_ignore_forbidden_fields_and_are_fold_bounded():
    frame = pd.DataFrame({'SMILES': ['C', 'CC', 'CCC', 'N', 'CN', 'CO'],
                          'Title': list('abcdef'), 'strain': ['x']*6, 'Culture': ['Vero']*6})
    model = PowerRegression('sqrt50-log50', n_estimators=10, seeds=(0,)).fit(frame, np.array([0, 1, 2, 3, 4, 36066.667]))
    first = model.predict(frame)
    frame['CC50_uM'] = 1e10
    frame['IC50_uM_VACV'] = 0
    frame['SI VACV'] = 1e10
    np.testing.assert_array_equal(first, model.predict(frame))
    assert np.isfinite(first).all() and (first >= 0).all() and (first <= 36066.667).all()
    assert model.maximum_ == 36066.667


@pytest.mark.parametrize('y', [[-1], [np.nan], []])
def test_invalid_targets_rejected(y):
    with pytest.raises(ValueError):
        PowerRegression().fit(pd.DataFrame(index=range(len(y))), y)
