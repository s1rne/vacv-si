"""Fixed molecular Bayesian bootstrap of v4 Ridge-log pair regression."""
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
from experiments.v4_pair_delta import training_pairs, fit_predict, antisymmetric_predict
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = ("bag-raw", "bag-log", "bag-raw-half-v4")
DRAWS = 16
RNG_SEED = 5100


def molecule_pair_weights(pairs_local, train_size):
    """One positive draw per molecule, shared across every incident pair."""
    rng = np.random.default_rng(RNG_SEED)
    molecule_weights = rng.exponential(size=(DRAWS, train_size))
    weights = np.sqrt(molecule_weights[:, pairs_local[:, 0]] * molecule_weights[:, pairs_local[:, 1]])
    return weights / weights.mean(axis=1, keepdims=True)


def fit_bags(x, y, similarity, matching, tr, va):
    pairs = training_pairs(similarity, matching, tr)
    baseline, fired, _ = fit_predict("ridge-log", x, y, similarity, matching, tr, va)
    sim, match = similarity[np.ix_(va, tr)], matching[np.ix_(va, tr)]
    fallback = analog(sim, np.zeros_like(match), y[tr], "context-log70")[0]
    anchors = []
    for k in range(len(va)):
        ix = np.flatnonzero(match[k] & (sim[k] >= .7))
        anchors.append(ix[np.argsort(-sim[k, ix], kind="stable")[:3]])
    draws = np.tile(fallback, (DRAWS, 1))
    if len(pairs):
        local = {int(row): i for i, row in enumerate(tr)}
        pair_local = np.array([[local[int(a)], local[int(b)]] for a, b in pairs])
        transformed = np.log1p(y[tr])
        delta = x[pairs[:, 0]] - x[pairs[:, 1]]
        target = transformed[pair_local[:, 0]] - transformed[pair_local[:, 1]]
        for draw, weights in enumerate(molecule_pair_weights(pair_local, len(tr))):
            model = Ridge(alpha=10., fit_intercept=False).fit(delta, target, sample_weight=weights)
            for k, ix in enumerate(anchors):
                if not len(ix):
                    continue
                changes = antisymmetric_predict(model, x[va[k]][None, :] - x[tr[ix]])
                values = np.expm1(transformed[ix] + changes)
                anchor_weights = sim[k, ix] ** 8
                draws[draw, k] = max(0., float(anchor_weights @ values / anchor_weights.sum()))
    raw = draws.mean(axis=0)
    predictions = {"bag-raw": raw, "bag-log": np.expm1(np.log1p(draws).mean(axis=0)),
                   "bag-raw-half-v4": .5 * (raw + baseline), "v4-ridge-log": baseline}
    if not all(np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("Nonfinite bootstrap predictions")
    audit = {"directed_pairs": len(pairs), "expert_rows": int(fired.sum()),
             "anchor_counts": [len(a) for a in anchors],
             "anchors": [[int(tr[a]) for a in ix] for ix in anchors],
             "draw_prediction_min": draws.min(axis=0).tolist(),
             "draw_prediction_max": draws.max(axis=0).tolist(),
             "draw_prediction_std": draws.std(axis=0).tolist()}
    return predictions, audit


def main():
    out = Path("experiments/results/v5-pair-bagging")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    data = load(Path("data/raw_v2"), require_v2=True)
    x, y = fingerprint(data.train), data.y
    similarity = tanimoto(x, x)
    ctx = contexts(data.train)
    matching = same_context(ctx, ctx)
    metrics, audits, winner = [], [], None
    with threadpool_limits(limits=2):
        for split in (41, 42, 43, 44, 45, "scaffold"):
            names = PROFILES if split == 41 else (winner,)
            predictions = {n: np.full(len(y), np.nan) for n in (*names, "v4-ridge-log")}
            folds, seen = np.full(len(y), -1), np.zeros(len(y), dtype=int)
            if split == "scaffold":
                groups = scaffold_folds(scaffolds_of(data.train), 5)
                iterator = (fold_indices(groups, f) for f in range(5))
            else:
                iterator = KFold(5, shuffle=True, random_state=split).split(y)
            for fold, (tr, va) in enumerate(iterator):
                values, audit = fit_bags(x, y, similarity, matching, tr, va)
                for name in predictions:
                    predictions[name][va] = values[name]
                folds[va], seen[va] = fold, seen[va] + 1
                audits.append({"split": split, "fold": fold, "train_rows": tr.tolist(),
                               "validation_rows": va.tolist(), **audit})
                print(f"split={split} fold={fold} done", flush=True)
            assert len(y) == 320 and (seen == 1).all()
            for name, pred in predictions.items():
                metrics.append({"split": split, "model": name, **diagnostics(y, pred)})
            if split == 41:
                winner = min((m for m in metrics if m["model"] in PROFILES), key=lambda m: m["mse"])["model"]
                (out / "frozen-winner.json").write_text(json.dumps({"winner": winner, "discovery": 41, "profiles": PROFILES}, indent=2) + "\n")
            pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds, **predictions}).to_csv(out / f"oof-{split}.csv", index=False)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(metrics[-len(predictions):], flush=True)
    selected = [m for m in metrics if m["model"] == winner and m["split"] in (42, 43, 44, 45)]
    controls = [m for m in metrics if m["model"] == "v4-ridge-log" and m["split"] in (42, 43, 44, 45)]
    manifest = {"winner": winner, "profiles": PROFILES, "discovery": 41, "confirmation": [42, 43, 44, 45],
                "draws": DRAWS, "rng_seed": RNG_SEED, "alpha": 10., "weighting": "sqrt(w_i*w_j), normalized mean one per draw; exponential molecule weights",
                "all_train_molecules_eligible_as_anchors": True, "external_data": False,
                "all_labels_retained": True, "rows": len(y), "target_max": float(y.max()),
                "confirmation_mean_mse": float(np.mean([m["mse"] for m in selected])),
                "control_mean_mse": float(np.mean([m["mse"] for m in controls])),
                "improves_each_confirmation": all(a["mse"] < b["mse"] for a, b in zip(selected, controls)),
                "upload_3m_each_gate": all(m["mse"] <= 3e6 for m in selected), "submitted": False,
                "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path("data/raw_v2/train_v2.csv"), Path("experiments/v4_pair_delta.py"),
                            Path("experiments/v2_analog.py"), Path("experiments/v2_context_analog.py"))},
                "oof_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("oof-*.csv"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "support-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    lines = ["# Molecule-weighted pair bagging", "", f"Frozen winner: {winner}.", "",
             "| Split | Model | Full MSE |", "|---|---|---:|"]
    lines += [f"| {m['split']} | {m['model']} | {m['mse']:.3f} |" for m in metrics]
    lines += ["", json.dumps({k: v for k, v in manifest.items() if k not in ("hashes", "oof_hashes")}, indent=2), "",
              "Sixteen fixed exponential-weight draws per training molecule; pair weights are geometric means, normalized to mean one. "
              "Pairs, anchors, Ridge alpha 10, log delta target, antisymmetry and fallback exactly follow v4. "
              "All outer-training molecules remain anchor-eligible in every draw. Raw mean, log1p mean, and raw mean blended 50/50 with v4 "
              "were predeclared; discovery alone selected the frozen profile. Other profiles are not scored on confirmation. "
              "Every fold retains all labels including the maximum. No upper prediction caps or external data. "
              "Support audit reports anchors and between-draw spread. This repeated CV is not independent public-score evidence. No submission created or uploaded.", "",
              "Run from repository root: `PYTHONPATH=. python experiments/v5_pair_bagging.py` (output directory must be absent).",
              "Tests: `PYTHONPATH=. python -m pytest -q tests/test_v5_pair_bagging.py`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
