"""Frozen train-only matched-context molecular pair difference experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from experiments.v2_context_analog import contexts, same_context, predict as analog
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = ("ridge-raw", "ridge-log", "et-raw", "et-log")


def training_pairs(similarity, matching, train_indices):
    """Return global endpoint indices, restricted to the outer training set.

    Each undirected pair is selected if either endpoint has the other in its
    top eight same-context neighbors. Both orientations have equal weight.
    No target values enter selection.
    """
    allowed = np.asarray(train_indices, dtype=int)
    edges = set()
    for i in allowed:
        js = allowed[(allowed != i) & matching[i, allowed] & (similarity[i, allowed] >= .5)]
        js = js[np.argsort(-similarity[i, js], kind="stable")[:8]]
        edges.update((min(i, j), max(i, j)) for j in js)
    pairs = [(i, j) for i, j in sorted(edges) for i, j in ((i, j), (j, i))]
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def antisymmetric_predict(model, differences):
    return .5 * (model.predict(differences) - model.predict(-differences))


def fit_predict(profile, x, y, similarity, matching, tr, va):
    pairs = training_pairs(similarity, matching, tr)
    sim = similarity[np.ix_(va, tr)]
    match = matching[np.ix_(va, tr)]
    result = analog(sim, np.zeros_like(match), y[tr], "context-log70")[0]
    fired = np.zeros(len(va), dtype=bool)
    if not len(pairs):
        return result, fired, 0
    transformed = np.log1p(y[tr]) if profile.endswith("log") else y[tr] / 1000.
    # Restrict label access explicitly to outer training labels.
    local = {int(row): i for i, row in enumerate(tr)}
    a = np.array([local[int(i)] for i in pairs[:, 0]])
    b = np.array([local[int(i)] for i in pairs[:, 1]])
    delta = x[pairs[:, 0]] - x[pairs[:, 1]]
    target = transformed[a] - transformed[b]
    model = (Ridge(alpha=10., fit_intercept=False) if profile.startswith("ridge") else
             ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3,
                                 random_state=330, n_jobs=2))
    model.fit(delta, target)
    for k, row in enumerate(va):
        anchors = np.flatnonzero(match[k] & (sim[k] >= .7))
        anchors = anchors[np.argsort(-sim[k, anchors], kind="stable")[:3]]
        if not len(anchors):
            continue
        changes = antisymmetric_predict(model, x[row][None, :] - x[tr[anchors]])
        values = transformed[anchors] + changes
        values = np.expm1(values) if profile.endswith("log") else values * 1000.
        weights = sim[k, anchors] ** 8
        result[k] = max(0., float(weights @ values / weights.sum()))
        fired[k] = True
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite predictions: {profile}")
    return result, fired, len(pairs)


def main():
    out = Path("experiments/results/v4-pair-delta")
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
        for split in (33, 34, 35, "scaffold"):
            names = PROFILES if split == 33 else (winner,)
            iterator = (KFold(5, shuffle=True, random_state=split).split(y) if split != "scaffold"
                        else (fold_indices(scaffold_folds(scaffolds_of(data.train), 5), f) for f in range(5)))
            predictions = {name: np.full(len(y), np.nan) for name in (*names, "context-raw70")}
            gates = {name: np.zeros(len(y), dtype=bool) for name in names}
            folds = np.full(len(y), -1)
            for fold, (tr, va) in enumerate(iterator):
                folds[va] = fold
                predictions["context-raw70"][va] = analog(sim[np.ix_(va, tr)], match[np.ix_(va, tr)], y[tr], "context-raw70")[0]
                for name in names:
                    predictions[name][va], gates[name][va], n = fit_predict(name, x, y, sim, match, tr, va)
                    audits.append({"split": split, "fold": fold, "profile": name, "directed_pairs": n,
                                   "train_rows": tr.tolist(), "validation_rows": va.tolist()})
                print(f"split={split} fold={fold} completed", flush=True)
            for name, prediction in predictions.items():
                if not np.isfinite(prediction).all():
                    raise ValueError(name)
                metrics.append({"split": split, "model": name, **diagnostics(y, prediction),
                                "pair_expert_rows": int(gates[name].sum()) if name in gates else None})
            pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds,
                          **predictions, **{f"{n}-gate": g for n, g in gates.items()}}).to_csv(out/f"oof-{split}.csv", index=False)
            (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            if split == 33:
                winner = min((m for m in metrics if m["model"] in PROFILES), key=lambda m: m["mse"])["model"]
                (out/"frozen-winner.json").write_text(json.dumps({"winner": winner, "discovery_seed": 33})+"\n")
            print(metrics[-len(predictions):], flush=True)
    selected = [m for m in metrics if m["model"] == winner and m["split"] in (34, 35)]
    bases = [m for m in metrics if m["model"] == "context-raw70" and m["split"] in (34, 35)]
    mean = float(np.mean([m["mse"] for m in selected]))
    improves = all(a["mse"] < b["mse"] for a, b in zip(selected, bases))
    manifest = {"winner": winner, "discovery": 33, "confirmation": [34, 35], "profiles": PROFILES,
                "confirmation_mean_mse": mean, "improves_both_paired": improves,
                "local_4m_gate": mean < 4e6 and improves, "upload_3m_gate": all(m["mse"] <= 3e6 for m in selected),
                "external_data": False, "forbidden_features": False, "all_labels_retained": True,
                "label_count": len(y), "max_target": float(y.max()),
                "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path("data/raw_v2/train_v2.csv"), Path("experiments/v2_analog.py"),
                            Path("experiments/v2_context_analog.py"))}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out/"fold-audit.json").write_text(json.dumps(audits, indent=2)+"\n")
    lines = ["# Train-only molecular pair differences", "", f"Frozen discovery winner: `{winner}`.", "",
             "| Split | Model | Full raw MSE |", "|---|---|---:|"]
    lines += [f"| {m['split']} | {m['model']} | {m['mse']:.3f} |" for m in metrics]
    lines += ["", f"Confirmation mean: {mean:.3f}; improves both paired: {improves}.",
              f"4M local gate: {manifest['local_4m_gate']}; 3M each upload gate: {manifest['upload_3m_gate']}.",
              "", "All 320 labels retained. Same-context pairs selected using structures only inside each outer training fold; "
              "both orientations included. Morgan bit differences; Ridge alpha=10 or ET 400 trees leaf=3; raw/1000 or log1p deltas. "
              "Top-8 pair neighbors at similarity >=0.5, top-3 prediction anchors >=0.7, similarity^8 weighting; "
              "structural log-KNN fallback. Nonnegative final outputs, no upper cap. "
              "Identical allowed representations stay in their predeclared random folds; scaffold is a separate stress test. "
              "These are local CV results, not independent leaderboard evidence. No submission was generated or uploaded."]
    (out/"report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
