"""Query-local weighted pair-delta Ridge versus frozen global v4."""
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
from experiments.v4_pair_delta import training_pairs, antisymmetric_predict, fit_predict as baseline_predict
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = {"local4": 4, "local8": 8, "half-local4": 4}
BASELINE = "v4-ridge-log"


def local_weights(query_similarity, pairs, power):
    if not len(pairs):
        return np.array([], dtype=float)
    weights = .1 + .9 * np.maximum(query_similarity[pairs[:, 0]], query_similarity[pairs[:, 1]]) ** power
    return weights / weights.mean()


def fit_predict(profile, x, y, similarity, matching, tr, va):
    if profile == "half-local4":
        return .5 * (fit_predict("local4", x, y, similarity, matching, tr, va) +
                     baseline_predict("ridge-log", x, y, similarity, matching, tr, va)[0])
    power = PROFILES[profile]
    pairs = training_pairs(similarity, matching, tr)
    sim, match = similarity[np.ix_(va, tr)], matching[np.ix_(va, tr)]
    result = analog(sim, np.zeros_like(match), y[tr], "context-log70")[0]
    if not len(pairs):
        return result
    local = {int(row): i for i, row in enumerate(tr)}
    transformed = np.log1p(y[tr])
    a = np.array([local[int(i)] for i in pairs[:, 0]])
    b = np.array([local[int(i)] for i in pairs[:, 1]])
    delta_x = x[pairs[:, 0]] - x[pairs[:, 1]]
    # Zero training columns have exactly zero Ridge coefficients.
    active = np.any(delta_x != 0, axis=0)
    delta_x = delta_x[:, active]
    target = transformed[a] - transformed[b]
    cache = {}
    for k, row in enumerate(va):
        anchors = np.flatnonzero(match[k] & (sim[k] >= .7))
        anchors = anchors[np.argsort(-sim[k, anchors], kind="stable")[:3]]
        if not len(anchors):
            continue
        pair_weights = local_weights(similarity[row], pairs, power)
        key = pair_weights.tobytes()
        if key not in cache:
            model = Ridge(alpha=10., fit_intercept=False)
            if active.any():
                model.fit(delta_x, target, sample_weight=pair_weights)
                cache[key] = model
            else:
                cache[key] = None
        model = cache[key]
        differences = (x[row][None, :] - x[tr[anchors]])[:, active]
        delta = antisymmetric_predict(model, differences) if model is not None else np.zeros(len(anchors))
        values = np.expm1(transformed[anchors] + delta)
        weights = sim[k, anchors] ** 8
        result[k] = max(0., float(weights @ values / weights.sum()))
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite {profile}")
    return result


def main():
    out = Path("experiments/results/v6-pair-local")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    data = load(Path("data/raw_v2"), require_v2=True)
    x, y = fingerprint(data.train), data.y
    sim = tanimoto(x, x)
    ctx = contexts(data.train)
    match = same_context(ctx, ctx)
    metrics, audit, winner = [], [], None
    with threadpool_limits(limits=2):
        for split in (51, 52, 53, 54, 55, "scaffold"):
            names = tuple(PROFILES) if split == 51 else (winner,)
            iterator = (KFold(5, shuffle=True, random_state=split).split(y) if split != "scaffold" else
                        (fold_indices(scaffold_folds(scaffolds_of(data.train), 5), f) for f in range(5)))
            predictions = {name: np.full(len(y), np.nan) for name in (*names, BASELINE)}
            folds = np.full(len(y), -1)
            for fold, (tr, va) in enumerate(iterator):
                folds[va] = fold
                predictions[BASELINE][va] = baseline_predict("ridge-log", x, y, sim, match, tr, va)[0]
                for name in names:
                    if name == "half-local4" and "local4" in predictions:
                        predictions[name][va] = .5 * (predictions["local4"][va] + predictions[BASELINE][va])
                    else:
                        predictions[name][va] = fit_predict(name, x, y, sim, match, tr, va)
                audit.append({"split": split, "fold": fold, "train_rows": tr.tolist(), "validation_rows": va.tolist()})
            for name, prediction in predictions.items():
                metrics.append({"split": split, "model": name, **diagnostics(y, prediction)})
            pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds, **predictions}).to_csv(out/f"oof-{split}.csv", index=False)
            (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            if split == 51:
                winner = min((m for m in metrics if m["model"] in PROFILES), key=lambda m: m["mse"])["model"]
                (out/"frozen-winner.json").write_text(json.dumps({"winner": winner, "discovery_seed": 51})+"\n")
            print(metrics[-len(predictions):], flush=True)
    selected = [m for m in metrics if m["split"] in (52, 53, 54, 55) and m["model"] == winner]
    base = [m for m in metrics if m["split"] in (52, 53, 54, 55) and m["model"] == BASELINE]
    means = {winner: float(np.mean([m["mse"] for m in selected])), BASELINE: float(np.mean([m["mse"] for m in base]))}
    manifest = {"profiles": PROFILES, "discovery": 51, "confirmation": [52, 53, 54, 55], "winner": winner,
                "confirmation_mean_mse": means, "paired_wins": sum(a["mse"] < b["mse"] for a, b in zip(selected, base)),
                "all_labels_retained": True, "label_count": len(y), "max_target": float(y.max()),
                "external_data": False, "test_data_used": False, "repeated_cv_not_independent_data": True,
                "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path("experiments/v4_pair_delta.py"), Path("experiments/v2_analog.py"),
                            Path("experiments/v2_context_analog.py"), Path("data/raw_v2/train_v2.csv"))}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out/"fold-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    lines = ["# Query-local pair Ridge", "", f"Discovery51 frozen winner: `{winner}`.", "",
             "| Split | Model | Full raw MSE |", "|---|---|---:|"]
    lines += [f"| {m['split']} | {m['model']} | {m['mse']:.3f} |" for m in metrics]
    lines += ["", f"Confirmation means: {json.dumps(means)}; paired wins: {manifest['paired_wins']}/4.",
              "", "Three predeclared profiles: local power4, local power8, 50/50 local4 plus v4. Validation uses the unchanged v4 Ridge-log baseline on identical folds. "
              "Per-query mean-one pair weights: 0.1+0.9*max(sim(query,a),sim(query,b))**power. Query labels never used. Alpha=10; fingerprints, pair selection and query anchors unchanged. Ridge is refitted locally using outer-training pairs only. "
              "All 320 labels retained, no cap/external data/forbidden concentration features. "
              "Repeated CV is not independent data, and local scores are not comparable numerically with public score. "
              "No submission generated or uploaded."]
    (out/"report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
