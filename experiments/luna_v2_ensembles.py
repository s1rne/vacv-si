"""Audit aligned official-v2 OOFs and freeze bounded blends on seed 9 only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results/luna-v2-ensembles"
SOURCES = {
    "et_log": ("v2-power", "log"),
    "et_power": ("v2-power", "sqrt50-log50"),
    "tweedie": ("v2-tweedie", "tweedie-1.8"),
    "tabicl": ("v2-tabicl", "tabicl-log"),
    "analog": ("v2-analog", "knn8-log"),
    "hurdle": ("v2-hurdle", "hurdle-0"),
}
SPLITS = ("9", "10", "11", "scaffold")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_oof(directory: str, split: str) -> pd.DataFrame:
    path = ROOT / f"experiments/results/{directory}/oof-{split}.csv"
    frame = pd.read_csv(path)
    id_col = "index" if "index" in frame else "row"
    out = frame.rename(columns={id_col: "row"})
    if "target" not in out:
        y = pd.read_csv(ROOT / "data/raw_v2/train_v2.csv")["SI VACV"].to_numpy()
        out["target"] = y[out.row.to_numpy(dtype=int)]
    return out


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"Output directory already exists: {RESULTS}")
    data = pd.read_csv(ROOT / "data/raw_v2/train_v2.csv")
    if len(data) != 320 or not {"SMILES", "Title", "strain", "Culture"}.issubset(data):
        raise ValueError("Unexpected official-v2 training schema/row count")
    y = data["SI VACV"].to_numpy(dtype=float)
    if not np.isfinite(y).all() or y.max() != 36066.667:
        raise ValueError("Official-v2 full target vector mismatch")

    frames: dict[str, dict[str, pd.DataFrame]] = {}
    for split in SPLITS:
        loaded = {name: read_oof(directory, split) for name, (directory, _) in SOURCES.items()}
        base = loaded["et_log"]
        for name, frame in loaded.items():
            if (len(frame) != len(y) or not np.array_equal(frame.row.to_numpy(), base.row.to_numpy())
                    or not np.array_equal(frame.fold.to_numpy(), base.fold.to_numpy())
                    or not np.array_equal(frame.target.to_numpy(), y)):
                raise ValueError(f"OOF alignment mismatch: {split}/{name}")
        frames[split] = loaded

    # Optimize convex source weights on discovery seed 9 only. All later files
    # use these fixed weights without adjustment.
    names = list(SOURCES)
    x9 = np.column_stack([frames["9"][name][column].to_numpy(dtype=float)
                          for name, (_, column) in SOURCES.items()])
    objective = lambda w: float(np.mean((x9 @ w - y) ** 2))
    fit = minimize(objective, np.full(len(names), 1 / len(names)), method="SLSQP",
                   bounds=[(0, 1)] * len(names),
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 3000, "ftol": 1e-10})
    if not fit.success or not np.isfinite(fit.x).all() or (fit.x < -1e-8).any():
        raise RuntimeError(f"Discovery-only convex optimization failed: {fit.message}")
    optimized = np.maximum(fit.x, 0)
    optimized /= optimized.sum()

    # Six predeclared candidate families; ET is the exact paired reference.
    weights: dict[str, dict[str, float]] = {
        "et_log_baseline": {"et_log": 1.0},
        "et_power_50": {"et_log": .5, "et_power": .5},
        "et_tweedie_50": {"et_log": .5, "tweedie": .5},
        "et_tabicl_50": {"et_log": .5, "tabicl": .5},
        "et_analog_50": {"et_log": .5, "analog": .5},
        "optimized_all_sources_seed9": dict(zip(names, optimized.tolist())),
    }

    metrics = []
    oof = {}
    for split, source_frames in frames.items():
        matrix = np.column_stack([source_frames[name][column].to_numpy(dtype=float)
                                  for name, (_, column) in SOURCES.items()])
        prediction = {}
        for candidate, blend in weights.items():
            w = np.array([blend.get(name, 0.0) for name in names])
            p = matrix @ w
            if not np.isfinite(p).all() or (p < -1e-8).any():
                raise ValueError(f"Invalid blend output: {split}/{candidate}")
            prediction[candidate] = p
            metrics.append({"split": split, "candidate": candidate,
                            "mse": float(np.mean((p-y)**2)),
                            "mse_delta_vs_et": float(np.mean((p-y)**2)-np.mean((matrix[:, 0]-y)**2)),
                            "max_label_prediction": float(p[int(np.argmax(y))]),
                            "max_prediction": float(p.max())})
        oof[split] = pd.DataFrame({"row": np.arange(len(y)), "target": y,
                                   "fold": source_frames["et_log"].fold.to_numpy(),
                                   **prediction})

    discovery = [row for row in metrics if row["split"] == "9"]
    selected = min(discovery, key=lambda row: row["mse"])["candidate"]
    confirms = [row for row in metrics if row["candidate"] == selected and row["split"] in ("10", "11")]
    scaffold = next(row for row in metrics if row["candidate"] == selected and row["split"] == "scaffold")
    gate = len(confirms) == 2 and all(row["mse"] <= 3_000_000 for row in confirms)

    RESULTS.mkdir(parents=True)
    for split, frame in oof.items():
        frame.to_csv(RESULTS / f"oof-{split}.csv", index=False)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifests = {}
    for directory in sorted({directory for directory, _ in SOURCES.values()}):
        for path in sorted((ROOT / f"experiments/results/{directory}").glob("*.json")):
            manifests[str(path.relative_to(ROOT))] = digest(path)
    source_paths = [ROOT / f"experiments/results/{d}/oof-{s}.csv"
                    for d in sorted({d for d, _ in SOURCES.values()}) for s in SPLITS]
    report = {
        "target_version": "train_v2", "rows": len(y), "target_max": float(y.max()),
        "train_sha256": digest(ROOT / "data/raw_v2/train_v2.csv"),
        "source_sha256": digest(Path(__file__)), "oof_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        "source_manifest_sha256": manifests, "sources": SOURCES,
        "mapping_audit": "all source OOFs match positional row IDs, all 320 targets, and fold IDs for each split",
        "blend_weights": weights, "weights_optimized_on": "seed 9 only; SLSQP nonnegative simplex",
        "selected_on_discovery9": selected,
        "confirmation10_11": confirms, "scaffold": scaffold,
        "passes_both_confirmation_3m_gate": gate,
        "candidate_csv": None,
        "kaggle_upload": False,
    }
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    baseline = [r for r in metrics if r["candidate"] == "et_log_baseline"]
    improvement = {s: next(r["mse"] for r in metrics if r["split"] == s and r["candidate"] == selected)
                   - next(r["mse"] for r in baseline if r["split"] == s) for s in SPLITS}
    baseline_mse = {row["split"]: row["mse"] for row in baseline}
    selected_mse = {row["split"]: row["mse"] for row in metrics if row["candidate"] == selected}
    delta_pct = {s: 100 * improvement[s] / baseline_mse[s] for s in SPLITS}
    lines = ["# Luna v2 OOF ensemble audit", "",
             "All comparisons use identical positional rows, full targets and fold IDs across the saved source OOFs. The paired ET reference is `v2-power` `log` (the same three-seed ExtraTrees context/structure pipeline); no seed-6/7/8 baseline is mixed into these comparisons.", "",
             f"Seed 9 selected `{selected}`. Frozen confirmation MSE: seed 10 {confirms[0]['mse']:.3f}; seed 11 {confirms[1]['mse']:.3f}; scaffold {scaffold['mse']:.3f}.",
             "Paired full raw-MSE comparison (all 320 rows):",
             "", "| Split | ET log baseline | Frozen blend | Delta | Change |",
             "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        lines.append(f"| {split} | {baseline_mse[split]:,.3f} | {selected_mse[split]:,.3f} | {improvement[split]:+,.3f} | {delta_pct[split]:+.3f}% |")
    lines += [
             f"3,000,000 confirmation gate: {'PASS' if gate else 'FAIL'}. No test candidate CSV was produced and no Kaggle upload was made.", "",
             "All 320 rows and the 36,066.667 target are retained; only SMILES, Title, strain and Culture are used as features by the source experiments. The ensemble is a convex average of saved predictions, with weights fitted on seed 9 only. No test labels or public scores were used. The train-only prediction cap implies a 3.016M lower bound from the held-out maximum alone; these convex blends therefore cannot satisfy the 3M gate.", "",
             "No candidate CSV was generated because the frozen candidate fails the gate. No Kaggle upload was made.", "",
             "Reproduce from the repository root with the existing environment:", "",
             "```sh",
             "DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si-novel/.venv/lib/python3.13/site-packages/sklearn/.dylibs PYTHONPATH=. /Users/mike/Projects/vacv-si-novel/.venv/bin/python experiments/luna_v2_ensembles.py",
             "```", "",
             "See `metrics.json`, `report.json`, and the four complete OOF files for source hashes and predictions."]
    (RESULTS / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"selected": selected, "weights": weights[selected], "confirmations": confirms,
                      "scaffold": scaffold, "passes_gate": gate, "paired_delta": improvement}, indent=2))


if __name__ == "__main__":
    main()
