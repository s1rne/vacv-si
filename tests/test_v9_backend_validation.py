import numpy as np
import pytest

from experiments import v9_backend_validation as runner


def test_group_drift_rejected_even_if_fold_assignments_match(monkeypatch):
    ids = np.asarray(["a", "b"])
    groups = np.asarray([0, 1])
    frozen = {"identities": ids.tolist(), "scaffold_groups": groups.tolist(), "folds": {}}
    monkeypatch.setattr(runner, "make_folds", lambda *args: {})
    runner.verify_groups(frozen, ids, groups)
    with pytest.raises(ValueError, match="frozen protocol"):
        runner.verify_groups(frozen, np.asarray(["changed", "b"]), groups)
    with pytest.raises(ValueError, match="frozen protocol"):
        runner.verify_groups(frozen, ids, np.asarray([3, 4]))
