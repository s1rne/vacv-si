"""Fixed anchor threshold experiment; v4 training and representation unchanged."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v4_pair_delta import training_pairs, antisymmetric_predict, fit_predict as v4_predict
from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from experiments.v2_context_analog import contexts, same_context, predict as analog
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = {"gate55": .55, "gate60": .60, "gate65": .65}
BASELINE = "v4-gate70"
OUT = Path("experiments/results/v5-pair-anchor-gate")


def fit_predict(threshold, x, y, similarity, matching, tr, va):
    """Exact v4 ridge-log mechanics except the query anchor threshold."""
    pairs = training_pairs(similarity, matching, tr)
    sim, match = similarity[np.ix_(va, tr)], matching[np.ix_(va, tr)]
    result = analog(sim, np.zeros_like(match), y[tr], "context-log70")[0]
    fired = np.zeros(len(va), bool)
    if not len(pairs):
        return result, fired, 0
    target = np.log1p(y[tr])
    local = {int(row): i for i, row in enumerate(tr)}
    a = np.array([local[int(i)] for i in pairs[:, 0]])
    b = np.array([local[int(i)] for i in pairs[:, 1]])
    model = Ridge(alpha=10., fit_intercept=False).fit(x[pairs[:, 0]] - x[pairs[:, 1]], target[a] - target[b])
    for k, row in enumerate(va):
        anchors = np.flatnonzero(match[k] & (sim[k] >= threshold))
        anchors = anchors[np.argsort(-sim[k, anchors], kind="stable")[:3]]
        if not len(anchors):
            continue
        changes = antisymmetric_predict(model, x[row][None, :] - x[tr[anchors]])
        values = np.expm1(target[anchors] + changes)
        weights = sim[k, anchors] ** 8
        result[k] = max(0., float(weights @ values / weights.sum()))
        fired[k] = True
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite predictions")
    return result, fired, len(pairs)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    y, x = data.y, fingerprint(data.train)
    sim, ctx = tanimoto(x, x), contexts(data.train)
    match = same_context(ctx, ctx)
    metrics, audits, winner = [], [], None
    for split in (46, 47, 48, 49, 50, "scaffold"):
        names = (*PROFILES, BASELINE) if split == 46 else (winner, BASELINE)
        if split == "scaffold":
            groups = scaffold_folds(scaffolds_of(data.train), 5)
            iterator = (fold_indices(groups, f) for f in range(5))
        else:
            iterator = KFold(5, shuffle=True, random_state=split).split(y)
        predictions = {n: np.full(len(y), np.nan) for n in names}
        fired = {n: np.zeros(len(y), bool) for n in names}
        folds = np.full(len(y), -1)
        for fold, (tr, va) in enumerate(iterator):
            folds[va] = fold
            pairs = training_pairs(sim, match, tr)
            assert not np.isin(pairs, va).any()
            audits.append({"split": split, "fold": fold, "train_rows": tr.tolist(),
                           "validation_rows": va.tolist(), "directed_pairs": len(pairs),
                           "pair_rows": np.unique(pairs).tolist()})
            for name in names:
                p, gate, _ = (v4_predict("ridge-log", x, y, sim, match, tr, va) if name == BASELINE
                              else fit_predict(PROFILES[name], x, y, sim, match, tr, va))
                predictions[name][va], fired[name][va] = p, gate
            print(f"split={split} fold={fold} done", flush=True)
        for name, p in predictions.items():
            assert np.isfinite(p).all()
            metrics.append({"split": split, "model": name, **diagnostics(y, p),
                            "pair_expert_rows": int(fired[name].sum()),
                            "max_label_expert": bool(fired[name][np.argmax(y)])})
        pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds,
                      **predictions, **{n + "-fired": fired[n] for n in names}}).to_csv(OUT / f"oof-{split}.csv", index=False)
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if split == 46:
            winner = min((m for m in metrics if m["model"] in PROFILES), key=lambda m: m["mse"])["model"]
            (OUT / "frozen-winner.json").write_text(json.dumps({"winner": winner, "discovery_seed": 46}) + "\n")
        print(json.dumps(metrics[-len(names):]), flush=True)
    confirm = [m for m in metrics if m["split"] in (47, 48, 49, 50) and m["model"] == winner]
    base = [m for m in metrics if m["split"] in (47, 48, 49, 50) and m["model"] == BASELINE]
    avg, avgbase = float(np.mean([m["mse"] for m in confirm])), float(np.mean([m["mse"] for m in base]))
    manifest = {"winner": winner, "profiles": PROFILES, "discovery": 46, "confirmations": [47, 48, 49, 50],
                "confirmation_mean_mse": avg, "paired_v4_mean_mse": avgbase,
                "improves_all_confirmations": all(m["mse"] < b["mse"] for m, b in zip(confirm, base)),
                "each_confirmation_below_3m": all(m["mse"] <= 3e6 for m in confirm),
                "rows": len(y), "max_target": float(y.max()),
                "external_data": False, "forbidden_features": False, "submission_created": False,
                "seed45_not_evaluated_or_used_for_selection": True,
                "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__), Path("experiments/v4_pair_delta.py"),
                            Path("experiments/v2_analog.py"), Path("experiments/v2_context_analog.py"),
                            Path("vacv/pipeline.py"), Path("vacv/features.py"), Path("data/raw_v2/train_v2.csv")]}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "fold-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    lines = ["# Pair-delta query anchor gate", "", f"Frozen seed46 winner: {winner}.", "",
             "| Split | Gate | Full raw MSE | Expert rows |", "|---|---|---:|---:|"]
    lines += [f"| {m['split']} | {m['model']} | {m['mse']:.3f} | {m['pair_expert_rows']} |" for m in metrics]
    lines += ["", f"Confirmation mean: {avg:.3f}; paired v4 mean: {avgbase:.3f}; relative change: {(avg/avgbase-1)*100:.3f}%.",
              f"Improves every confirmation: {manifest['improves_all_confirmations']}. Each confirmation <=3M: {manifest['each_confirmation_below_3m']}.", "",
              "Only query anchor similarity threshold changes: fixed .55/.60/.65 versus v4 .70. Training pairs, Morgan radius2 binary2048, Ridge alpha10 no intercept, log1p deltas, top3 anchors, similarity^8 weights, structural-log fallback unchanged. All320 labels retained without clipping/removal, full raw MSE. Seed45 was hypothesis motivation only and is not evaluated or used in selection. Discovery46 chooses among exactly three profiles; frozen winner confirmed on47–50 with scaffold stress. No external data, forbidden fields, CSV or upload. Paired baseline is v4 Ridge-log, not the weaker context-only model.", ""]
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
