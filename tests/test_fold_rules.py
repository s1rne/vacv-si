"""Правила подстановки не должны подбираться по контрольным строкам фолда."""

import numpy as np
import pandas as pd

from vacv.literature import LOOKUP, fold_fit, fold_table, lookup


def test_full_fit_reproduces_the_shipped_table():
    saved = pd.read_csv(LOOKUP, low_memory=False)
    rebuilt = fold_fit()
    for column in ("substitute_recommended", "si_recommended", "si_blended", "sentinel_rate_for_bucket"):
        assert rebuilt[column].equals(saved[column]), column


def test_withholding_a_bucket_withdraws_its_recommendation():
    """Бакет, у которого не осталось обучающих строк, не рекомендуется к подстановке."""
    full = fold_fit()
    train = full[full["split"].eq("train")].reset_index(drop=True)
    recommended = full["substitute_recommended"].fillna(False).to_numpy(dtype=bool)
    bucket = train.loc[train.index[recommended[: len(train)]][0], ["match_tier", "rule"]]
    keep = np.flatnonzero(
        ~(train["match_tier"].eq(bucket["match_tier"]) & train["rule"].eq(bucket["rule"])).to_numpy()
    )
    withheld = fold_fit(keep)
    same_bucket = (full["match_tier"].eq(bucket["match_tier"]) & full["rule"].eq(bucket["rule"])).to_numpy()
    assert recommended[same_bucket].any()
    assert not withheld["substitute_recommended"].fillna(False).to_numpy(dtype=bool)[same_bucket].any()


def test_fold_table_has_one_row_per_molecule():
    """Таблица для модели совпадает по форме с обычной: иначе слияние размножит строки."""
    table, shipped = fold_table(), lookup()
    assert list(table.columns) == list(shipped.columns)
    assert not table["smiles"].duplicated().any()
    assert table.reset_index(drop=True).equals(shipped.reset_index(drop=True))


def test_zero_inactive_removes_the_inactive_term_but_not_hard_literature():
    import numpy as np
    from vacv.literature import to_zero_inactive

    frame = pd.DataFrame({"literature_si": [np.nan, np.nan], "literature_quality": [None, None]})
    out = to_zero_inactive(np.array([3000.0, 20.0]), np.array([0.2, 0.0]), frame)
    assert out.tolist() == [1000.0, 20.0]


def test_loader_maps_train_v2_zeros_back_to_the_internal_sentinel(tmp_path):
    from vacv.dataset import FORBIDDEN, load

    base = pd.DataFrame({"SMILES": ["C", "CC", "CCC"], "strain": ["a"] * 3, "Culture": ["c"] * 3,
                         "SI VACV": [0.0, 5.0, 7.0], FORBIDDEN[0]: [1, 2, 3], FORBIDDEN[1]: [0, 1, 1]})
    base.to_csv(tmp_path / "train.csv", index=False)
    base.to_csv(tmp_path / "test.csv", index=False)
    assert load(tmp_path).y.tolist() == [10000.0, 5.0, 7.0]
