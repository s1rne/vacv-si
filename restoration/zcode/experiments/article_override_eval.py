"""Эффект хард-оверрайдов из article_si.csv на экранах (сохранённые OOF-вероятности).

Гибридная вероятность p = (context + kernel) / 2, A(x) из ActiveValueModel,
soft lookup — как в проде; поверх — переопределения из таблиц статей:
sentinel → 10000, известный SI → точное значение.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.doc_si_extract import unit_scale  # noqa: F401
from experiments.stacking import fold_arrays, SOURCES
from experiments.stacking_eval import load_sources
from vacv.dataset import load
from vacv.literature import attach
from experiments.active_value2 import attach_nn
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds
from vacv.article_lookup import article_frame


def eval_split(pieces, overrides: pd.DataFrame | None) -> float:
    sq = n = 0
    for piece in pieces:
        p = (piece["probs"]["context"] + piece["probs"]["kernel"]) / 2
        raw = p * 10000.0 + (1 - p) * piece["active"]
        prediction = raw.copy()
        found = np.isfinite(piece["lit"])
        prediction[found] = p[found] * 10000.0 + (1 - p[found]) * piece["lit"][found]
        if overrides is not None:
            valid = piece["valid"]
            fr = piece["frame"]
            mask_sent = fr["article_sentinel"].to_numpy(dtype=bool)
            mask_si = fr["article_si"].notna().to_numpy()
            prediction[mask_sent] = 10000.0
            prediction[mask_si] = fr.loc[mask_si, "article_si"].to_numpy(dtype=float)
        sq += float(np.sum((prediction - piece["y"]) ** 2))
        n += len(piece["y"])
    return sq / n


def main() -> None:
    data = load()
    frame, y = attach_nn(attach(data.train)), data.y
    art = article_frame()
    frame = frame.reset_index(drop=True)
    frame["article_sentinel"] = frame["SMILES"].map(art.set_index("smiles")["sentinel"]).fillna(False)
    frame["article_si"] = frame["SMILES"].map(art.set_index("smiles")["si"])

    splits = {f"random-{s}": [v for _, v in KFold(5, shuffle=True, random_state=s).split(y)]
              for s in (0, 1, 2, 3, 4, 5)}
    splits["scaffold"] = scaffold_folds(scaffolds_of(frame), n_splits=5)

    results = []
    for name in ("random-0", "random-1", "random-2"):
        probs, group = load_sources(Path("outputs/stacking-screen"), name)
        pieces = fold_arrays(probs, group, frame, y, splits[name])
        for piece in pieces:
            piece["frame"] = frame.iloc[piece["valid"]]
        base = eval_split(pieces, None)
        over = eval_split(pieces, art)
        results.append({"split": name, "baseline": base, "with_article": over})
        print(f"{name}: baseline {base:,.0f} → with article {over:,.0f} ({(over - base) / base:+.2%})",
              flush=True)
    mean_b = float(np.mean([r["baseline"] for r in results]))
    mean_o = float(np.mean([r["with_article"] for r in results]))
    print(f"screen mean: {mean_b:,.0f} → {mean_o:,.0f} ({(mean_o - mean_b) / mean_b:+.2%})")

    for name in ("random-3", "random-4", "random-5", "scaffold"):
        probs, group = load_sources(Path("outputs/stacking-confirm"), name)
        pieces = fold_arrays(probs, group, frame, y, splits[name])
        for piece in pieces:
            piece["frame"] = frame.iloc[piece["valid"]]
        base = eval_split(pieces, None)
        over = eval_split(pieces, art)
        print(f"{name}: baseline {base:,.0f} → with article {over:,.0f} ({(over - base) / base:+.2%})",
              flush=True)


if __name__ == "__main__":
    main()
