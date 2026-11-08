"""Predeclared log-target offset ablation of unchanged v4 pair regression."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from experiments.v2_context_analog import contexts, same_context, predict as analog
from experiments.v4_pair_delta import training_pairs, antisymmetric_predict, fit_predict as baseline
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = {"offset-0.01": .01, "offset-0.1": .1, "offset-10": 10.}


def transform(y, offset):
    if offset <= 0 or np.any(np.asarray(y) < 0):
        raise ValueError("Positive offset and nonnegative targets required")
    return np.log(np.asarray(y) + offset)


def inverse(z, offset):
    return np.exp(z) - offset


def fit_predict(offset, x, y, similarity, matching, tr, va):
    pairs = training_pairs(similarity, matching, tr)
    sim, match = similarity[np.ix_(va, tr)], matching[np.ix_(va, tr)]
    result = analog(sim, np.zeros_like(match), y[tr], "context-log70")[0]
    anchors_audit = [[] for _ in va]
    if len(pairs):
        transformed = transform(y[tr], offset)
        local = {int(row): i for i, row in enumerate(tr)}
        a = np.array([local[int(i)] for i in pairs[:, 0]])
        b = np.array([local[int(i)] for i in pairs[:, 1]])
        delta = x[pairs[:, 0]] - x[pairs[:, 1]]
        target = transformed[a] - transformed[b]
        model = Ridge(alpha=10., fit_intercept=False).fit(delta, target)
        for k, row in enumerate(va):
            anchors = np.flatnonzero(match[k] & (sim[k] >= .7))
            anchors = anchors[np.argsort(-sim[k, anchors], kind="stable")[:3]]
            anchors_audit[k] = tr[anchors].tolist()
            if not len(anchors):
                continue
            changes = antisymmetric_predict(model, x[row][None, :] - x[tr[anchors]])
            values = inverse(transformed[anchors] + changes, offset)
            weights = sim[k, anchors] ** 8
            result[k] = max(0., float(weights @ values / weights.sum()))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite offset predictions")
    return result, {"directed_pairs": len(pairs), "anchors": anchors_audit,
                    "expert_rows": sum(bool(a) for a in anchors_audit)}


def main():
    out = Path("experiments/results/v6-pair-offset")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    data = load(Path("data/raw_v2"), require_v2=True)
    x, y = fingerprint(data.train), data.y
    sim = tanimoto(x, x)
    ctx = contexts(data.train)
    match = same_context(ctx, ctx)
    metrics, audits, winner = [], [], None
    with threadpool_limits(limits=2):
        for split in (56, 57, 58, 59, 60, "scaffold"):
            names = tuple(PROFILES) if split == 56 else (winner,)
            predictions = {n: np.full(len(y), np.nan) for n in (*names, "v4-ridge-log")}
            folds, seen = np.full(len(y), -1), np.zeros(len(y), dtype=int)
            if split == "scaffold":
                groups = scaffold_folds(scaffolds_of(data.train), 5)
                iterator = (fold_indices(groups, f) for f in range(5))
            else:
                iterator = KFold(5, shuffle=True, random_state=split).split(y)
            for fold, (tr, va) in enumerate(iterator):
                predictions["v4-ridge-log"][va] = baseline("ridge-log", x, y, sim, match, tr, va)[0]
                for name in names:
                    predictions[name][va], audit = fit_predict(PROFILES[name], x, y, sim, match, tr, va)
                    audits.append({"split": split, "fold": fold, "profile": name,
                                   "train_rows": tr.tolist(), "validation_rows": va.tolist(), **audit})
                folds[va], seen[va] = fold, seen[va] + 1
            assert len(y) == 320 and (seen == 1).all()
            for name, pred in predictions.items():
                metrics.append({"split": split, "model": name, **diagnostics(y, pred)})
            if split == 56:
                winner = min((m for m in metrics if m["model"] in PROFILES), key=lambda m: m["mse"])["model"]
                (out / "frozen-winner.json").write_text(json.dumps({"winner": winner, "discovery": 56, "profiles": PROFILES}, indent=2) + "\n")
            pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds, **predictions}).to_csv(out / f"oof-{split}.csv", index=False)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(metrics[-len(predictions):], flush=True)
    chosen = [m for m in metrics if m["model"] == winner and m["split"] in (57, 58, 59, 60)]
    controls = [m for m in metrics if m["model"] == "v4-ridge-log" and m["split"] in (57, 58, 59, 60)]
    manifest = {"winner": winner, "profiles": PROFILES, "discovery": 56, "confirmation": [57, 58, 59, 60],
                "transform": "log(y+c), inverse exp(z)-c; structural fallback unchanged log1p",
                "selection": "minimum full raw MSE on discovery only", "external_data": False,
                "all_labels_retained": True, "rows": len(y), "target_max": float(y.max()),
                "confirmation_mean_mse": float(np.mean([m["mse"] for m in chosen])),
                "control_mean_mse": float(np.mean([m["mse"] for m in controls])),
                "improves_each_confirmation": all(a["mse"] < b["mse"] for a, b in zip(chosen, controls)),
                "upload_3m_each_gate": all(m["mse"] <= 3e6 for m in chosen), "submitted": False,
                "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path("data/raw_v2/train_v2.csv"), Path("experiments/v4_pair_delta.py"),
                            Path("experiments/v2_analog.py"), Path("experiments/v2_context_analog.py"))},
                "oof_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("oof-*.csv"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "support-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    lines = ["# Log-target offset in molecular pair regression", "", f"Frozen winner: {winner}.", "",
             "| Split | Model | Full MSE |", "|---|---|---:|"]
    lines += [f"| {m['split']} | {m['model']} | {m['mse']:.3f} |" for m in metrics]
    lines += ["", "```json", json.dumps({k: v for k, v in manifest.items() if k not in ("hashes", "oof_hashes")}, indent=2), "```", "",
              "Fixed offsets c=0.01,0.1,10 modify only pair target and anchor transforms. All other v4 behavior remains: "
              "Ridge alpha10, both pair orientations, antisymmetric difference predictions, same-context top3 anchors >=0.7, "
              "raw-space aggregation and unchanged structural log1p fallback. Floor at zero only after anchor aggregation; "
              "no upper cap. All 320 labels including zeros and maximum retained. No external or forbidden features. "
              "Discovery chooses once; four confirmation splits and scaffold use the frozen winner and unchanged v4 control. "
              "No submission generated or uploaded. Repeated CV is not independent leaderboard evidence.", "",
              "Run: `PYTHONPATH=. python experiments/v6_pair_offset.py` (output must be absent).",
              "Tests: `PYTHONPATH=. python -m pytest -q tests/test_v6_pair_offset.py`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
