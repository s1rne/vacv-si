"""Custom PADRE-inspired background-aware molecular pair trees, not an exact reproduction."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from experiments.v2_context_analog import contexts, same_context, predict as analog
from experiments.v4_pair_delta import training_pairs, antisymmetric_predict, fit_predict as baseline_predict
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = {"leaf2": 2, "leaf5": 5}
BASELINE = "v4-ridge-log"
SOURCE = "https://wenhao-gao.github.io/assets/publications/2021_PADRE_JCIM/paper.pdf"


def pair_features(a, b):
    return np.concatenate((a-b, .5*(a+b)), axis=1)


def antisymmetric_background(model, a, b):
    return .5*(model.predict(pair_features(a, b))-model.predict(pair_features(b, a)))


def fit_predict(profile, x, y, similarity, matching, tr, va):
    pairs = training_pairs(similarity, matching, tr)
    sim, match = similarity[np.ix_(va, tr)], matching[np.ix_(va, tr)]
    result = analog(sim, np.zeros_like(match), y[tr], "context-log70")[0]
    if not len(pairs):
        return result
    local = {int(row): i for i, row in enumerate(tr)}
    transformed = np.log1p(y[tr])
    a = np.array([local[int(i)] for i in pairs[:, 0]])
    b = np.array([local[int(i)] for i in pairs[:, 1]])
    model = ExtraTreesRegressor(n_estimators=400, max_features=.5,
                                min_samples_leaf=PROFILES[profile], random_state=560, n_jobs=2)
    model.fit(pair_features(x[pairs[:, 0]], x[pairs[:, 1]]), transformed[a]-transformed[b])
    query_rows, anchor_rows, slices = [], [], []
    for k, row in enumerate(va):
        anchors = np.flatnonzero(match[k] & (sim[k] >= .7))
        anchors = anchors[np.argsort(-sim[k, anchors], kind="stable")[:3]]
        if not len(anchors):
            continue
        start = len(query_rows)
        query_rows.extend([row]*len(anchors))
        anchor_rows.extend(tr[anchors])
        slices.append((k, anchors, start, len(query_rows)))
    if query_rows:
        changes = antisymmetric_background(model, x[query_rows], x[anchor_rows])
        for k, anchors, start, end in slices:
            values = np.expm1(transformed[anchors]+changes[start:end])
            weights = sim[k, anchors]**8
            result[k] = max(0., float(weights@values/weights.sum()))
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite {profile}")
    return result


def main():
    out = Path("experiments/results/v6-pair-background")
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
        for split in (56, 57, 58, 59, 60, "scaffold"):
            names = tuple(PROFILES) if split == 56 else (winner,)
            iterator = (KFold(5, shuffle=True, random_state=split).split(y) if split != "scaffold" else
                        (fold_indices(scaffold_folds(scaffolds_of(data.train), 5), f) for f in range(5)))
            predictions = {name: np.full(len(y), np.nan) for name in (*names, BASELINE)}
            folds = np.full(len(y), -1)
            for fold, (tr, va) in enumerate(iterator):
                folds[va] = fold
                predictions[BASELINE][va] = baseline_predict("ridge-log", x, y, sim, match, tr, va)[0]
                for name in names:
                    predictions[name][va] = fit_predict(name, x, y, sim, match, tr, va)
                audit.append({"split": split, "fold": fold, "train_rows": tr.tolist(), "validation_rows": va.tolist()})
            print(f"split={split} all folds complete", flush=True)
            for name, prediction in predictions.items():
                metrics.append({"split": split, "model": name, **diagnostics(y, prediction)})
            pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds, **predictions}).to_csv(out/f"oof-{split}.csv", index=False)
            (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            if split == 56:
                winner = min((m for m in metrics if m["model"] in PROFILES), key=lambda m: m["mse"])["model"]
                (out/"frozen-winner.json").write_text(json.dumps({"winner": winner, "discovery_seed": 56})+"\n")
            print(metrics[-len(predictions):], flush=True)
    selected = [m for m in metrics if m["split"] in (57, 58, 59, 60) and m["model"] == winner]
    base = [m for m in metrics if m["split"] in (57, 58, 59, 60) and m["model"] == BASELINE]
    means = {winner: float(np.mean([m["mse"] for m in selected])), BASELINE: float(np.mean([m["mse"] for m in base]))}
    manifest = {"profiles": PROFILES, "discovery": 56, "confirmation": [57, 58, 59, 60], "winner": winner,
                "confirmation_mean_mse": means, "paired_wins": sum(a["mse"] < b["mse"] for a, b in zip(selected, base)),
                "all_labels_retained": True, "label_count": len(y), "max_target": float(y.max()),
                "source_paper": SOURCE, "exact_paper_reproduction": False, "external_data": False, "test_data_used": False, "repeated_cv_not_independent_data": True,
                "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path("experiments/v4_pair_delta.py"), Path("experiments/v2_analog.py"),
                            Path("experiments/v2_context_analog.py"), Path("data/raw_v2/train_v2.csv"))}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out/"fold-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    lines = ["# Background-aware pair trees", "", f"Discovery56 frozen winner: `{winner}`.", "",
             "| Split | Model | Full raw MSE |", "|---|---|---:|"]
    lines += [f"| {m['split']} | {m['model']} | {m['mse']:.3f} |" for m in metrics]
    lines += ["", f"Confirmation means: {json.dumps(means)}; paired wins: {manifest['paired_wins']}/4.",
              "", "Two predeclared ExtraTrees profiles: 400 trees, max_features=0.5, min_samples_leaf=2 or 5. Validation uses the unchanged v4 Ridge-log baseline on identical folds. "
              "Pair features concatenate fingerprint difference and mean; predictions antisymmetrized while retaining the mean background. Pair selection, fallback and query anchors unchanged. Custom PADRE-inspired variant, not a reproduction of its all-pairs training protocol. Source: https://doi.org/10.1021/acs.jcim.1c00670 . "
              "All 320 labels retained, no cap/external data/forbidden concentration features. "
              "Repeated CV is not independent data, and local scores are not comparable numerically with public score. "
              "No submission generated or uploaded."]
    (out/"report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
