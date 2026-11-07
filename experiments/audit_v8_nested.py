"""Independent saved-artifact audit; no imports from the fitting/selection code."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    out = Path("experiments/results/v8-nested-validation")
    manifest = json.loads((out / "manifest.json").read_text())
    protocol_bytes = (out / "protocol.json").read_bytes()
    assert protocol_bytes == Path("docs/v8-nested-protocol.json").read_bytes()
    assert hashlib.sha256(protocol_bytes).hexdigest() == manifest["protocol_sha256"]
    protocol = json.loads(protocol_bytes)
    for path, checksum in protocol["sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == checksum, path
    for path, checksum in manifest["artifacts_sha256"].items():
        assert hashlib.sha256((out / path).read_bytes()).hexdigest() == checksum, path
    names = protocol["config"]["models"]
    units = protocol["config"]["simplex_units"]
    grid = np.asarray([w for w in itertools.product(range(units, -1, -1), repeat=len(names))
                       if sum(w) == units], float) / units
    np.testing.assert_array_equal(grid, protocol["weight_grid"])
    y = pd.read_csv("data/raw_v2/train_v2.csv", usecols=["SI VACV"])["SI VACV"].to_numpy()
    identities = np.asarray(protocol["identity_groups"])
    components = np.asarray(protocol["scaffold_groups"])
    selection_rows = json.loads((out / "selection-audit.json").read_text())
    selections = {(a["scheme"], a["fold"]): a for a in selection_rows}
    assert len(selections) == len(selection_rows) == 20
    metrics = json.loads((out / "metrics.json").read_text())
    assert len(metrics) == 24
    metric_map = {(m["split"], m["model"]): m for m in metrics}
    assert len(metric_map) == len(metrics)
    checked_groups, checked_metrics, comparisons = 0, 0, []
    for scheme, folds in protocol["folds"].items():
        outer = pd.read_csv(out / f"oof-{scheme}.csv")
        np.testing.assert_array_equal(outer.row, np.arange(len(y)))
        np.testing.assert_allclose(outer.target, y, rtol=0, atol=1e-10)
        covered = np.zeros(len(y), int)
        for fold, indices in enumerate(folds):
            tr, va = np.asarray(indices["train"]), np.asarray(indices["valid"])
            covered[va] += 1
            np.testing.assert_array_equal(outer.fold.iloc[va], fold)
            inner_coverage = np.zeros(len(y), int)
            for f in indices["inner"]:
                assert set(f["train"]) | set(f["valid"]) == set(tr)
                inner_coverage[f["valid"]] += 1
            np.testing.assert_array_equal(np.flatnonzero(inner_coverage), np.sort(tr))
            np.testing.assert_array_equal(inner_coverage[tr], 1)
            for f in [indices, *indices["inner"]]:
                a, b = f["train"], f["valid"]
                assert not set(a) & set(b)
                assert not set(identities[a]) & set(identities[b])
                if scheme == "scaffold":
                    assert not set(components[a]) & set(components[b])
                checked_groups += 1
            inner = pd.read_csv(out / f"inner-{scheme}-{fold}.csv")
            np.testing.assert_array_equal(inner.row, tr)
            np.testing.assert_allclose(inner.target, y[tr], rtol=0, atol=1e-10)
            p = inner[names].to_numpy()
            assert np.isfinite(p).all() and (p >= 0).all()
            scores = ((p @ grid.T - y[tr, None]) ** 2).mean(axis=0)
            singles = ((p - y[tr, None]) ** 2).mean(axis=0)
            s = selections[scheme, fold]
            np.testing.assert_allclose(scores, s["grid_mse"], rtol=1e-12)
            np.testing.assert_allclose(singles, s["single_mse"], rtol=1e-12)
            np.testing.assert_array_equal(grid[np.argmin(scores)], s["weights"])
            assert int(np.argmin(singles)) == s["single_index"]
            q = outer.iloc[va][names].to_numpy()
            np.testing.assert_allclose(outer["nested-blend"].iloc[va], q @ np.asarray(s["weights"]), rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(outer["nested-single"].iloc[va], q[:, s["single_index"]], rtol=1e-12, atol=1e-10)
        np.testing.assert_array_equal(covered, 1)
        for name in (*names, "nested-single", "nested-blend"):
            p = outer[name].to_numpy()
            assert np.isfinite(p).all() and (p >= 0).all()
            mse = float(np.mean((p - y) ** 2))
            np.testing.assert_allclose(mse, metric_map[scheme, name]["mse"], rtol=1e-12)
            checked_metrics += 1
        comparisons.append({"scheme": scheme, **{n: metric_map[scheme, n]["mse"]
                            for n in ("v4", "nested-single", "nested-blend")}})
    old = pd.read_csv("experiments/results/v5-identity-stress/oof.csv")
    primary = pd.read_csv(out / "oof-identity.csv")
    np.testing.assert_array_equal(primary.fold, old.fold)
    np.testing.assert_allclose(primary.v4, old["ridge-log"], rtol=1e-12, atol=1e-10)
    report = {"passed": True, "protocol_sha256": manifest["protocol_sha256"],
              "inner_selections_recomputed": len(selections), "group_partitions_checked": checked_groups,
              "outer_metrics_recomputed": checked_metrics, "baseline_matches_prior_identity_stress": True,
              "comparisons": comparisons}
    (out / "independent-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
