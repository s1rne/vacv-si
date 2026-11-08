import numpy as np
from experiments.v3_source_pairs import interval, singleton, cache_activity, times


def test_bounds_are_not_exact_targets():
    ic = {'value_um': .8, 'relation': '>'}
    cc = {'value_um': 7.9, 'relation': '='}
    assert interval(ic, cc) == (0., 9.875, 'upper_bound')
    assert interval({'value_um': .8, 'relation': '='}, {'value_um': 7.9, 'relation': '>'}) == (9.875, np.inf, 'lower_bound')
    assert interval(ic, {'value_um': 7.9, 'relation': '>'}) is None


def test_mixed_records_never_become_median_exact():
    exact = {'activity_id': '1', 'assay_id': 'A', 'endpoint': 'CC50', 'value_um': 84.1, 'relation': '='}
    bound = dict(exact, activity_id='2', value_um=100, relation='>=')
    assert singleton([exact, bound]) is None
    assert singleton([exact, dict(exact, assay_id='B')]) is None
    assert singleton([exact, dict(exact, activity_id='2')])['value_um'] == 84.1


def test_cache_conflict_is_sticky():
    entries = {}
    original = {'activity_id': 1, 'standard_value': 84.1}
    for record in (original, dict(original, standard_value=100), original):
        cache_activity(entries, record, 'cache.json')
    assert entries['1']['conflict']
    assert entries['1']['record'] == original


def test_exposure_units_normalize():
    assert times('72 h exposure') == times('3 days exposure') == [72.]
