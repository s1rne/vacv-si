"""Аудит остаточных ошибок честного CV (art+paper+lit(fit)): топ-строки по
вкладу в MSE для прицельного литературного майнинга.

Протокол как lit_si_cv (правила внутри фолда), но сохраняет per-row предсказания
варианта +lit(fit) и печатает топ-ошибки с контекстом (Title, покрытие
article_frame, ChEMBL-доки).

Запуск: uv run python experiments/lit_error_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.lit_si_cv import apply_overrides, piece_mse, valid_lit_overrides
from experiments.lit_si_rule import (calibrate_pair_trusted,
                                     calibrate_sentinel_docs)
from experiments.stacking import fold_arrays
from experiments.stacking_eval import load_sources
from vacv.article_lookup import article_frame
from vacv.dataset import load
from vacv.literature import attach
from experiments.active_value2 import attach_nn
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds


def main() -> None:
    data = load()
    frame, y = attach_nn(attach(data.train)), np.asarray(data.y, dtype=float)
    frame = frame.reset_index(drop=True)

    art_fixed = pd.read_csv("data/processed/article_si.csv")
    paper = pd.read_csv("data/processed/paper_si.csv")
    bmcl = pd.read_csv("data/processed/bmcl_si.csv")
    rcb = pd.read_csv("data/processed/rcb_si.csv")
    art_all = pd.concat([art_fixed, paper, bmcl, rcb,
                         pd.read_csv("data/processed/cmdc_si.csv"),
                         pd.read_csv("data/processed/rsc2024_si.csv"),
                         pd.read_csv("data/processed/kovaleva_si.csv"),
                         pd.read_csv("data/processed/sid3dg_si.csv"),
                         pd.read_csv("data/processed/molecules2022_si.csv"),
                         pd.read_csv("data/processed/cbdv2018_si.csv"),
                         pd.read_csv("data/processed/av2021_si.csv"),
                         pd.read_csv("data/processed/retro2020_si.csv"),
                         pd.read_csv("data/processed/reframe2023_si.csv"),
                         pd.read_csv("data/processed/jmc2005_si.csv")],
                        ignore_index=True)
    art_all["Title"] = art_all["Title"].astype(str)
    art_all = art_all.drop_duplicates(["smiles", "Title"], keep="first")
    covered = set(art_all["smiles"])

    records = json.load(open("outputs/doc-scraper/records-by-key.json"))

    splits = {f"random-{s}": [v for _, v in KFold(5, shuffle=True, random_state=s).split(y)]
              for s in (0, 1, 2, 3, 4, 5)}
    splits["scaffold"] = scaffold_folds(scaffolds_of(frame), n_splits=5)

    oof = np.full(len(y), np.nan)
    for name in ("random-0", "random-1", "random-2"):
        run(name, Path("outputs/stacking-screen"), splits, frame, y,
            art_all, covered, records, oof)
    for name in ("random-3", "random-4", "random-5", "scaffold"):
        run(name, Path("outputs/stacking-confirm"), splits, frame, y,
            art_all, covered, records, oof)

    err = (oof - y) ** 2
    out = pd.DataFrame({"index": np.arange(len(y)), "Title": frame["Title"].astype(str),
                        "y": y, "oof": oof, "sq_err": err}).dropna()
    out.to_csv("outputs/doc-scraper/lit-error-audit.csv", index=False)
    top = out.sort_values("sq_err", ascending=False).head(25)
    art = article_frame().set_index("smiles")
    print("=== топ-25 вкладов в MSE (art+paper+lit(fit)):")
    total = out["sq_err"].sum()
    print(f"суммарный вклад: {total:,.0f} | средний MSE: {out['sq_err'].mean():,.0f}")
    for _, r in top.iterrows():
        s = data.train.iloc[int(r["index"])]["SMILES"] if hasattr(data.train, "iloc") else ""
        cov = art.loc[s] if s in art.index else None
        cov_s = f"covered({cov['pmc']}/{cov['label']}, si={cov['si']:.3f})" if cov is not None else "—"
        print(f"  #{int(r['index']):>3} '{r['Title']:<8}' y={r['y']:>9,.2f} oof={r['oof']:>9,.2f} "
              f"sq={r['sq_err']:>12,.0f} {cov_s}")


def run(name: str, src: Path, splits: dict, frame: pd.DataFrame, y: np.ndarray,
        art_all: pd.DataFrame, covered: set[str], records: dict, oof: np.ndarray) -> None:
    probs, group = load_sources(src, name)
    pieces = fold_arrays(probs, group, frame, y, splits[name])
    for piece in pieces:
        valid = piece["valid"]
        fit_mask = np.ones(len(y), dtype=bool)
        fit_mask[valid] = False
        sdocs = calibrate_sentinel_docs(records, frame.iloc[fit_mask].reset_index(drop=True),
                                        y[fit_mask])
        ptr = calibrate_pair_trusted(records, frame.iloc[fit_mask].reset_index(drop=True),
                                     y[fit_mask])
        lit = valid_lit_overrides(records, frame.iloc[valid], sdocs, ptr, covered)
        piece["frame"] = apply_overrides(frame.iloc[valid], art_all, lit)
        p = (piece["probs"]["context"] + piece["probs"]["kernel"]) / 2
        pred = p * 10000.0 + (1 - p) * piece["active"]
        found = np.isfinite(piece["lit"])
        pred[found] = p[found] * 10000.0 + (1 - p[found]) * piece["lit"][found]
        fr = piece["frame"]
        sent = fr["article_sentinel"].to_numpy(dtype=bool)
        si = fr["article_si"].to_numpy(dtype=float)
        pred[sent] = 10000.0
        known = np.isfinite(si)
        pred[known] = si[known]
        oof[valid] = pred


if __name__ == "__main__":
    main()
