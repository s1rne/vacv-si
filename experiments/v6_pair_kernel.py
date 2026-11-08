"""Predeclared nonlinear RKHS pair-difference regression, train-only v2."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from experiments.v2_context_analog import contexts, same_context, predict as analog
from experiments.v4_pair_delta import training_pairs, fit_predict as baseline
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = {f"pair-kernel-p{power}-a{alpha}": (power, alpha)
            for power in (1, 2) for alpha in (1, 10)}


def pair_gram(kernel, query_pairs, train_pairs):
    a, b = query_pairs[:, 0], query_pairs[:, 1]
    c, d = train_pairs[:, 0], train_pairs[:, 1]
    return kernel[np.ix_(a, c)] - kernel[np.ix_(a, d)] - kernel[np.ix_(b, c)] + kernel[np.ix_(b, d)]


def coefficients(kernel, pairs, target, alpha):
    # Both orientations duplicate the same squared residual. Dividing their
    # objective by two yields unique-pair ridge regularization alpha / 2.
    gram = pair_gram(kernel, pairs, pairs)
    return np.linalg.solve(gram + alpha / 2 * np.eye(len(pairs)), target)


def fit_predict(profile, y, similarity, matching, tr, va):
    power, alpha = PROFILES[profile]
    directed = training_pairs(similarity, matching, tr)
    pairs = directed[directed[:, 0] < directed[:, 1]]
    sim, match = similarity[np.ix_(va, tr)], matching[np.ix_(va, tr)]
    result = analog(sim, np.zeros_like(match), y[tr], "context-log70")[0]
    transformed = np.log1p(y[tr])
    local = {int(row): i for i, row in enumerate(tr)}
    anchors_audit = [[] for _ in va]
    if len(pairs):
        kernel = similarity ** power
        target = np.array([transformed[local[int(a)]] - transformed[local[int(b)]] for a, b in pairs])
        coef = coefficients(kernel, pairs, target, alpha)
        for k, row in enumerate(va):
            anchors = np.flatnonzero(match[k] & (sim[k] >= .7))
            anchors = anchors[np.argsort(-sim[k, anchors], kind="stable")[:3]]
            anchors_audit[k] = tr[anchors].tolist()
            if not len(anchors):
                continue
            query_pairs = np.column_stack([np.full(len(anchors), row), tr[anchors]])
            changes = pair_gram(kernel, query_pairs, pairs) @ coef
            values = np.expm1(transformed[anchors] + changes)
            weights = sim[k, anchors] ** 8
            result[k] = max(0., float(weights @ values / weights.sum()))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite kernel predictions")
    return result, {"unique_pairs": len(pairs), "directed_pairs": len(directed),
                    "anchors": anchors_audit, "expert_rows": sum(bool(a) for a in anchors_audit)}


def main():
    out = Path("experiments/results/v6-pair-kernel")
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
        for split in (51, 52, 53, 54, 55, "scaffold"):
            names = tuple(PROFILES) if split == 51 else (winner,)
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
                    predictions[name][va], audit = fit_predict(name, y, sim, match, tr, va)
                    audits.append({"split": split, "fold": fold, "profile": name,
                                   "train_rows": tr.tolist(), "validation_rows": va.tolist(), **audit})
                folds[va], seen[va] = fold, seen[va] + 1
                print(f"split={split} fold={fold} done", flush=True)
            assert len(y) == 320 and (seen == 1).all()
            for name, pred in predictions.items():
                metrics.append({"split": split, "model": name, **diagnostics(y, pred)})
            if split == 51:
                winner = min((m for m in metrics if m["model"] in PROFILES), key=lambda m: m["mse"])["model"]
                (out / "frozen-winner.json").write_text(json.dumps({"winner": winner, "discovery": 51, "profiles": PROFILES}, indent=2) + "\n")
            pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds, **predictions}).to_csv(out / f"oof-{split}.csv", index=False)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(metrics[-len(predictions):], flush=True)
    chosen = [m for m in metrics if m["model"] == winner and m["split"] in (52, 53, 54, 55)]
    controls = [m for m in metrics if m["model"] == "v4-ridge-log" and m["split"] in (52, 53, 54, 55)]
    manifest = {"winner": winner, "profiles": PROFILES, "discovery": 51, "confirmation": [52, 53, 54, 55],
                "regularization": "unique undirected pairs use alpha/2; equivalent to both orientations at alpha",
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
    lines = ["# Nonlinear RKHS pair differences", "", f"Frozen winner: {winner}.", "",
             "| Split | Model | Full MSE |", "|---|---|---:|"]
    lines += [f"| {m['split']} | {m['model']} | {m['mse']:.3f} |" for m in metrics]
    lines += ["", "```json", json.dumps({k: v for k, v in manifest.items() if k not in ("hashes", "oof_hashes")}, indent=2), "```", "",
              "Molecular kernel K=Tanimoto^power, power 1/2; pair kernel K(a,c)-K(a,d)-K(b,c)+K(b,d). "
              "Four profiles cross powers 1/2 with ridge alpha 1/10, chosen on seed 51 then frozen. "
              "Unique undirected pairs use alpha/2, equivalent to the doubled residual objective of both orientations at alpha. "
              "Antisymmetry follows directly from the pair kernel; no intercept. Pair selection, log1p targets, top3 anchors at similarity>=0.7, "
              "raw-space anchor aggregation, zero floor and structural log-KNN fallback exactly follow v4. "
              "All training labels retained; no upper prediction caps, external data or forbidden features. "
              "Held-out labels never enter pair targets, anchor values or any fit. Repeated CV is not independent leaderboard evidence. No submission created or uploaded.", "",
              "Run from repository root: `PYTHONPATH=. python experiments/v6_pair_kernel.py` (output directory must be absent).",
              "Tests: `PYTHONPATH=. python -m pytest -q tests/test_v6_pair_kernel.py`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
