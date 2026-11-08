import pandas as pd
import numpy as np
from experiments.v6_pair_metadata import representations


def test_title_features_do_not_depend_on_labels_or_other_rows():
    frame = pd.DataFrame({'SMILES':['CCO','CCN'],'Title':['8a-series','8b-series']})
    original = representations(frame)
    changed = representations(frame.assign(**{'SI VACV':[1e10,-1e10]}))
    single = representations(frame.iloc[:1])
    for key in original:
        np.testing.assert_array_equal(original[key],changed[key])
        np.testing.assert_array_equal(original[key][:1],single[key])
