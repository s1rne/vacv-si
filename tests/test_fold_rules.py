"""Правила подстановки не должны подбираться по контрольным строкам фолда."""

import numpy as np
import pandas as pd

from vacv.literature import LOOKUP, fold_table


def test_full_fit_reproduces_the_shipped_table():
    saved = pd.read_csv(LOOKUP, low_memory=False)
    rebuilt = fold_table()
    for column in ("substitute_recommended", "si_recommended", "si_blended", "sentinel_rate_for_bucket"):
        assert rebuilt[column].equals(saved[column]), column


def test_withholding_a_bucket_withdraws_its_recommendation():
    """Бакет, у которого не осталось обучающих строк, не рекомендуется к подстановке."""
    full = fold_table()
    train = full[full["split"].eq("train")].reset_index(drop=True)
    recommended = full["substitute_recommended"].fillna(False).to_numpy(dtype=bool)
    bucket = train.loc[train.index[recommended[: len(train)]][0], ["match_tier", "rule"]]
    keep = np.flatnonzero(
        ~(train["match_tier"].eq(bucket["match_tier"]) & train["rule"].eq(bucket["rule"])).to_numpy()
    )
    withheld = fold_table(keep)
    same_bucket = (full["match_tier"].eq(bucket["match_tier"]) & full["rule"].eq(bucket["rule"])).to_numpy()
    assert recommended[same_bucket].any()
    assert not withheld["substitute_recommended"].fillna(False).to_numpy(dtype=bool)[same_bucket].any()
