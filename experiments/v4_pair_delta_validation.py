"""Fixed repeated-CV stress checks of the frozen pair-delta Ridge-log winner.

No tuning, seed selection, candidate generation or Kaggle interaction.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from experiments.v2_context_analog import contexts, same_context, predict as analog
from experiments.v4_pair_delta import fit_predict
from vacv.dataset import load

SEEDS = (13, 14, 39, 40)
PROFILE = "ridge-log"


def main():
    out = Path("experiments/results/v4-pair-delta-validation")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    frozen = Path("experiments/results/v4-pair-delta/frozen-winner.json")
    if json.loads(frozen.read_text())["winner"] != PROFILE:
        raise ValueError("Frozen discovery winner changed")
    data = load(Path("data/raw_v2"), require_v2=True)
    x, y = fingerprint(data.train), data.y
    sim = tanimoto(x, x)
    ctx = contexts(data.train)
    match = same_context(ctx, ctx)
    metrics, audit = [], []
    with threadpool_limits(limits=2):
        for seed in SEEDS:
            pred = {name: np.full(len(y), np.nan) for name in (PROFILE, "context-raw70")}
            folds = np.full(len(y), -1)
            gate = np.zeros(len(y), dtype=bool)
            for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=seed).split(y)):
                folds[va] = fold
                pred[PROFILE][va], gate[va], count = fit_predict(PROFILE, x, y, sim, match, tr, va)
                pred["context-raw70"][va] = analog(sim[np.ix_(va, tr)], match[np.ix_(va, tr)], y[tr], "context-raw70")[0]
                audit.append({"seed": seed, "fold": fold, "directed_pairs": count,
                              "train_rows": tr.tolist(), "validation_rows": va.tolist()})
            for name, values in pred.items():
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite predictions: {seed}, {name}")
                metrics.append({"split": seed, "model": name, **diagnostics(y, values),
                                "pair_expert_rows": int(gate.sum()) if name == PROFILE else None})
            pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds,
                          **pred, "pair-expert": gate}).to_csv(out/f"oof-{seed}.csv", index=False)
            print(metrics[-2:], flush=True)
    means = {name: float(np.mean([m["mse"] for m in metrics if m["model"] == name]))
             for name in (PROFILE, "context-raw70")}
    paired = {str(seed): metrics[2*i+1]["mse"] - metrics[2*i]["mse"] for i, seed in enumerate(SEEDS)}
    source_paths = (Path(__file__), Path("experiments/v4_pair_delta.py"), Path("experiments/v2_analog.py"),
                    Path("experiments/v2_context_analog.py"), Path("data/raw_v2/train_v2.csv"), frozen)
    manifest = {"profile": PROFILE, "seeds": SEEDS, "mean_mse": means,
                "paired_improvement_positive_is_better": paired, "all_labels_retained": True,
                "label_count": len(y), "max_target": float(y.max()), "external_data": False,
                "repeated_cv_not_independent_data": True, "submissions_generated": False,
                "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    for name, value in (("metrics", metrics), ("manifest", manifest), ("fold-audit", audit)):
        (out/f"{name}.json").write_text(json.dumps(value, indent=2)+"\n")
    lines = ["# Frozen pair-delta validation", "", "Fixed Ridge-log; fixed seeds 13, 14, 39, 40. "
             "Repeated CV on the same 320 molecules, not independent data. No retuning or seed selection.", "",
             "| Seed | Ridge-log full raw MSE | Context-KNN full raw MSE | Improvement |", "|---|---:|---:|---:|"]
    for i, seed in enumerate(SEEDS):
        a, b = metrics[2*i]["mse"], metrics[2*i+1]["mse"]
        lines.append(f"| {seed} | {a:.3f} | {b:.3f} | {(b-a)/b*100:.3f}% |")
    lines += ["", f"All four means: pair={means[PROFILE]:.3f}, baseline={means['context-raw70']:.3f}.",
              "All labels retained, including maximum. Forbidden concentration columns and external data excluded. "
              "Complete OOF predictions and fold membership saved. No submission CSV generated or uploaded."]
    (out/"report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
