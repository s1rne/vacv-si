"""Nested подбор веса бленда (w_graph) и силы графовой поправки (beta).

Управляющий агент: полный построение правил внутри каждого фолда. Схема:
для каждого фолда fit=4/5 делится на sub-fit (80%) / sub-holdout (20%);
на sub-holdout перебирается сетка (w_graph — вес графа в бленде с CheMeleon-
головой, beta — сила correct_probability), выбирается комбинация с минимальным
MSE; затем модель переучивается на полном fit и оценивается на valid=1/5 с
выбранной комбинацией. Референс — дефолт (w_graph=0.75, beta=0.25).

Запуск: uv run python experiments/blend_weight_sweep.py --splits 3 4 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from threadpoolctl import threadpool_limits

from vacv.article_lookup import apply_article_overrides
from vacv.checkpoint import BestCheckpointModel, load_embeddings, SENTINEL
from vacv.dataset import load
from vacv.graph_correction import correct_probability, safe_features
from vacv.literature import attach
from vacv.pipeline import scaffolds_of
from vacv.postprocess import conditional_lookup
from vacv.splits import fold_indices, scaffold_folds
from experiments.pretrained_embeddings import predict_head

W_GRID = (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0)
BETA_GRID = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
DEFAULT_W, DEFAULT_BETA = 0.75, 0.25


def components_for(model, attached, test):
    """Компоненты продакшн-пути один раз на набор строк (TabICL/граф дорогие):
    негейтованная гибридная вероятность + дельты графа + химическая голова + AV."""
    query = safe_features(attached)
    with threadpool_limits(limits=4):
        graph_comps = model.graph_.components(attached, unlabeled=test)
        chem = np.asarray(predict_head(model.train_, model.y_, query, model.embedding_train_,
                                       model._embedding_matrix(query), "che-tabicl",
                                       model.checkpoint), dtype=float)
    return {"hybrid": np.asarray(graph_comps["probability"], dtype=float),
            "delta": np.asarray(graph_comps["deltas"]["diffusion"], dtype=float),
            "chem": chem,
            "av": np.asarray(model.active_values_.predict(attached), dtype=float)}


def predict_from(comp, attached, w_graph, beta):
    """Продакшн-путь с другими (w_graph, beta): бленд → AV → lookup → оверрайды."""
    graph = correct_probability(comp["hybrid"], comp["delta"], beta, uncertainty_gate=True)
    p = w_graph * graph + (1 - w_graph) * comp["chem"]
    raw = p * SENTINEL + (1 - p) * comp["av"]
    pred = conditional_lookup(raw, p, attached)
    return apply_article_overrides(attached, pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["3", "4", "5"])
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("data/raw/models/tabicl-classifier-v2-20260212.ckpt"))
    ap.add_argument("--features", type=Path, default=Path("data/raw/models/chemeleon_features.npz"))
    ap.add_argument("--out", type=Path, default=Path("outputs/doc-scraper/blend-sweep"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = load()
    frame, test = attach(data.train), attach(data.test)
    y = np.asarray(data.y, dtype=float)
    embeddings, _ = load_embeddings(args.features)

    rows = []
    for split in args.splits:
        folds = (scaffold_folds(scaffolds_of(frame), 5) if split == "scaffold" else
                 [v for _, v in KFold(5, shuffle=True, random_state=int(split)).split(y)])
        for fold in range(5):
            fit, valid = fold_indices(folds, fold)
            sub_fit, sub_hold = train_test_split(fit, test_size=0.2, random_state=7)
            sel = BestCheckpointModel(embeddings, args.checkpoint, active_value=True) \
                .fit(frame.iloc[sub_fit], y[sub_fit])
            comp_hold = components_for(sel, frame.iloc[sub_hold], test)
            scores = {}
            for w in W_GRID:
                for beta in BETA_GRID:
                    pred = predict_from(comp_hold, frame.iloc[sub_hold], w, beta)
                    scores[(w, beta)] = float(np.mean((pred - y[sub_hold]) ** 2))
            (w_star, beta_star) = min(scores, key=scores.get)
            full = BestCheckpointModel(embeddings, args.checkpoint, active_value=True) \
                .fit(frame.iloc[fit], y[fit])
            comp_valid = components_for(full, frame.iloc[valid], test)
            pred_star = predict_from(comp_valid, frame.iloc[valid], w_star, beta_star)
            pred_def = predict_from(comp_valid, frame.iloc[valid], DEFAULT_W, DEFAULT_BETA)
            mse_star = float(np.mean((pred_star - y[valid]) ** 2))
            mse_def = float(np.mean((pred_def - y[valid]) ** 2))
            rows.append({"split": split, "fold": fold, "w": w_star, "beta": beta_star,
                         "sel_mse": scores[(w_star, beta_star)], "mse": mse_star,
                         "mse_default": mse_def})
            print(f"{split} fold{fold}: w={w_star} beta={beta_star} | "
                  f"valid {mse_star:,.0f} vs default {mse_def:,.0f} "
                  f"({(mse_star - mse_def) / mse_def:+.2%})", flush=True)

    res = pd.DataFrame(rows)
    print("\n=== итог:", flush=True)
    print(f"выбранные w: {sorted(res['w'].unique())}, beta: {sorted(res['beta'].unique())}")
    print(f"valid MSE: nested {res['mse'].mean():,.0f} vs default {res['mse_default'].mean():,.0f} "
          f"({(res['mse'].mean() - res['mse_default'].mean()) / res['mse_default'].mean():+.2%})")
    res.to_csv(args.out / "sweep.csv", index=False)
    print(f"→ {args.out / 'sweep.csv'}", flush=True)


if __name__ == "__main__":
    main()
