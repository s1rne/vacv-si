"""Веса смеси источников вероятности неактивности.

Сейчас hybrid — равная смесь context_half и kernel, а чекпоинт добавляет CheMeleon
с фиксированным весом 0.25 и групповой долей не пользуется вовсе. Веса никогда не
искали совместно. Здесь каждый источник даёт OOF-вероятность на тех же фолдах,
после чего весовая сетка оценивается через полный постпроцессинг: смесь → A(x) →
условная литературная подстановка → MSE. Отбор по сплитам 0–2, подтверждение на 3–5.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.active_value2 import attach_nn, soft_lookup
from experiments.pretrained_embeddings import predict_head
from vacv.active_value import ActiveValueModel
from vacv.dataset import load
from vacv.inference import predict_ensemble
from vacv.literature import attach
from vacv.pipeline import INACTIVE, group_rate, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds

TABICL = Path("data/raw/models/tabicl-classifier-v2-20260212.ckpt")
SOURCES = ("context", "kernel", "grouped", "che")


def source_probabilities(frame: pd.DataFrame, y: np.ndarray, folds, seeds: int,
                         embeddings, checkpoint: Path) -> dict[str, np.ndarray]:
    out = {name: np.full(len(y), np.nan) for name in SOURCES}
    for fold in range(len(folds)):
        train, valid = fold_indices(folds, fold)
        tr, va = frame.iloc[train], frame.iloc[valid]
        out["context"][valid] = predict_ensemble(tr, y[train], va, seeds, "context_half")[0]
        out["kernel"][valid] = predict_ensemble(tr, y[train], va, seeds, "kernel")[0]
        out["grouped"][valid] = predict_ensemble(tr, y[train], va, seeds, "baseline")[0]
        emb_tr = np.stack([embeddings[s] for s in tr["SMILES"]])
        emb_va = np.stack([embeddings[s] for s in va["SMILES"]])
        out["che"][valid] = predict_head(tr, y[train], va, emb_tr, emb_va, "che-tabicl", checkpoint)
        print(f"fold {fold} done", flush=True)
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError("Incomplete source probabilities")
    return out


def group_rate_oof(frame: pd.DataFrame, y: np.ndarray, folds) -> np.ndarray:
    """Доля неактивных в группе по outer-train фолда — легитимный признак для валидации."""
    out = np.full(len(y), np.nan)
    labels = (y == INACTIVE).astype(float)
    for fold in range(len(folds)):
        train, valid = fold_indices(folds, fold)
        out[valid] = group_rate(frame.iloc[train], labels[train], frame.iloc[valid])
    return out


def fold_arrays(probs: dict[str, np.ndarray], group: np.ndarray,
                frame: pd.DataFrame, y: np.ndarray, folds) -> list[dict]:
    """На фолд: источники вероятности, A(x) и литература. Веса на них не влияют."""
    pieces = []
    for fold in range(len(folds)):
        train, valid = fold_indices(folds, fold)
        av = ActiveValueModel().fit(frame.iloc[train], y[train])
        literature = frame.iloc[valid]["literature_si"].to_numpy(dtype=float, na_value=np.nan)
        pieces.append({
            "valid": valid,
            "probs": {name: probs[name][valid] for name in probs},
            "group": group[valid],
            "active": av.predict(frame.iloc[valid]),
            "lit": np.where(np.isfinite(literature), np.clip(literature, 0, 10000), np.nan),
            "y": y[valid],
        })
    return pieces


def evaluate_blend(pieces: list[dict], weights: dict[str, float], group_weight: float) -> float:
    total = sum(weights.values()) + group_weight
    if total <= 0:
        raise ValueError("weights must not all be zero")
    sq = 0.0
    n = 0
    for piece in pieces:
        probability = sum(weights[name] * piece["probs"][name] for name in piece["probs"]) / total \
            + group_weight * piece["group"] / total
        raw = probability * 10000.0 + (1 - probability) * piece["active"]
        prediction = raw.copy()
        found = np.isfinite(piece["lit"])
        prediction[found] = probability[found] * 10000.0 + (1 - probability[found]) * piece["lit"][found]
        sq += float(np.sum((prediction - piece["y"]) ** 2))
        n += len(piece["y"])
    return sq / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--features", type=Path, default=Path("data/raw/models/chemeleon_features.npz"))
    parser.add_argument("--checkpoint", type=Path, default=TABICL)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/stacking"))
    args = parser.parse_args()
    import hashlib
    manifest = json.loads(args.features.with_suffix(".json").read_text())
    if hashlib.sha256(args.features.read_bytes()).hexdigest() != manifest["embedding_sha256"]:
        raise ValueError("Embedding cache checksum mismatch")
    with np.load(args.features, allow_pickle=False) as archive:
        embeddings = dict(zip(archive["smiles"], archive["embeddings"], strict=True))
    data = load()
    frame, y = attach_nn(attach(data.train)), data.y
    splits = {f"random-{s}": [v for _, v in KFold(5, shuffle=True, random_state=s).split(y)]
              for s in ((0, 1, 2) if args.screen_only else (3, 4, 5))}
    splits["scaffold"] = scaffold_folds(scaffolds_of(frame), n_splits=5)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, folds in splits.items():
        probs = source_probabilities(frame, y, folds, args.seeds, embeddings, args.checkpoint)
        group = group_rate_oof(frame, y, folds)
        np.savez(args.out / f"sources-{name}.npz", **probs, group=group)
        pieces = fold_arrays(probs, group, frame, y, folds)
        # Опорные точки: текущие фиксированные веса.
        references = {
            "hybrid(0.5/0.5)": ({"context": 1, "kernel": 1, "grouped": 0, "che": 0}, 0.0),
            "checkpoint(0.75che25)": ({"context": 0.375, "kernel": 0.375, "grouped": 0, "che": 0.25}, 0.0),
        }
        for label, (weights, gw) in references.items():
            mse = evaluate_blend(pieces, weights, gw)
            results.append({"split": name, "weights": label, "mse": mse})
            print(f"{name:12} {label:24} MSE={mse:,.0f}", flush=True)
        if name.startswith("random-"):
            best = None
            grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for wc, wk, wg, wche in itertools.product(grid, repeat=4):
                s = wc + wk + wg + wche
                if s == 0:
                    continue
                mse = evaluate_blend(pieces, {"context": wc, "kernel": wk, "grouped": wg, "che": wche}, 0.0)
                if best is None or mse < best[0]:
                    best = (mse, (wc, wk, wg, wche))
            # Добавка групповой доли к лучшей четверке.
            wc, wk, wg, wche = best[1]
            for gw in (0.05, 0.1, 0.2, 0.3, 0.5):
                mse = evaluate_blend(pieces, {"context": wc, "kernel": wk, "grouped": wg, "che": wche}, gw)
                if mse < best[0]:
                    best = (mse, (wc, wk, wg, wche, gw))
            results.append({"split": name, "weights": f"best{best[1]}", "mse": best[0]})
            print(f"{name:12} best={best[1]} MSE={best[0]:,.0f}", flush=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
