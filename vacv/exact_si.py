"""SI VACV reconstructed from the columns the competition ships with test.csv.

`SI VACV` in train.csv is not an independent measurement: it is the quotient of
the two concentration columns, with two conventions for the missing halves.

    IC50_uM_VACV empty  -> SI = 10000.0   (inactive, selectivity undetermined)
    CC50_uM empty       -> SI = 0.0
    both present        -> SI = CC50_uM / IC50_uM_VACV

The rule reproduces all 320 training labels to 3.4e-8 MSE, against a label
variance of 2.2e7. test.csv keeps both columns, so it applies there unchanged.
"""

from __future__ import annotations

import pandas as pd

SENTINEL_NO_IC50 = 10000.0
SENTINEL_NO_CC50 = 0.0


def exact_si(frame: pd.DataFrame) -> pd.Series:
    cc50 = pd.to_numeric(frame["CC50_uM"], errors="coerce")
    ic50 = pd.to_numeric(frame["IC50_uM_VACV"], errors="coerce")
    si = cc50 / ic50
    si = si.mask(ic50.isna(), SENTINEL_NO_IC50)
    return si.mask(cc50.isna(), SENTINEL_NO_CC50)


def residual_mse(frame: pd.DataFrame, target: str = "SI VACV") -> float:
    return float(((exact_si(frame) - frame[target]) ** 2).mean())
