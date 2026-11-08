"""Оценка фиксированных весов смеси на сохранённых OOF-источниках."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.stacking import fold_arrays, evaluate_blend
from vacv.dataset import load
from vacv.literature import attach
from experiments.active_value2 import attach_nn
from vacv.splits import fold_indices


def load_sources(out: Path, name: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    archive = np.load(out / f"sources-{name}.npz")
    group = archive["group"]
    probs = {k: archive[k] for k in archive.files if k != "group"}
    return probs, group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-out", type=Path, default=Path("outputs/stacking-screen"))
    parser.add_argument("--eval-out", type=Path, default=Path("outputs/stacking-confirm"))
    parser.add_argument("--splits", nargs="+", default=["3", "4", "5", "scaffold"])
    args = parser.parse_args()
    data = load()
    frame, y = attach_nn(attach(data.train)), data.y
    screen_names = ["random-0", "random-1", "random-2"]
    screen = [load_sources(args.screen_out, n) for n in screen_names]
    screen_pieces = [fold_arrays(p, g, frame, y,
                                 [v for _, v in __import__("sklearn.model_selection", fromlist=["KFold"])
                                  .KFold(5, shuffle=True, random_state=int(n.split("-")[1])).split(y)])
                     for (p, g), n in zip(screen, screen_names)]
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best = None
    for wc, wk, wg, wche in itertools.product(grid, repeat=4):
        if wc + wk + wg + wche == 0:
            continue
        mses = [evaluate_blend(pieces, {"context": wc, "kernel": wk, "grouped": wg, "che": wche}, 0.0)
                for pieces in screen_pieces]
        mean = float(np.mean(mses))
        if best is None or mean < best[0]:
            best = (mean, (wc, wk, wg, wche))
    print(f"screen-mean best weights {best[1]}: {best[0]:,.0f}", flush=True)
    for gw in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        mses = [evaluate_blend(pieces, {"context": best[1][0], "kernel": best[1][1],
                                        "grouped": best[1][2], "che": best[1][3]}, gw)
                for pieces in screen_pieces]
        mean = float(np.mean(mses))
        print(f"  +group {gw}: {mean:,.0f}", flush=True)
    weights = {"context": best[1][0], "kernel": best[1][1], "grouped": best[1][2], "che": best[1][3]}
    results = {"screen_best_weights": best[1], "screen_mean_mse": best[0], "eval": []}
    for name in args.splits:
        npz_name = name if name == "scaffold" else f"random-{name}"
        probs, group = load_sources(args.eval_out, npz_name)
        pieces = fold_arrays(probs, group, frame, y, None) if False else None
        # фолды пересчитываем по имени сплита
        from sklearn.model_selection import KFold
        from vacv.pipeline import scaffolds_of
        from vacv.splits import scaffold_folds
        folds = (scaffold_folds(scaffolds_of(frame), 5) if name == "scaffold" else
                 [v for _, v in KFold(5, shuffle=True, random_state=int(name)).split(y)])
        pieces = fold_arrays(probs, group, frame, y, folds)
        for label, w, gw in (("screen-best", weights, 0.0),
                             ("checkpoint-ref", {"context": 0.375, "kernel": 0.375, "grouped": 0, "che": 0.25}, 0.0)):
            mse = evaluate_blend(pieces, w, gw)
            results["eval"].append({"split": name, "weights": label, "mse": mse})
            print(f"{name:12} {label:16} MSE={mse:,.0f}", flush=True)
    (args.screen_out / "chosen-weights.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
