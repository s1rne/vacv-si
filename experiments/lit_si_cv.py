"""Честная CV-валидация оверрайдов: правила строятся ВНУТРИ каждого фолда.

Управляющий агент: новые правила проверять только на разрешённой целевой колонке
train, с полным построением правил внутри каждого фолда; цифра 3-4M (in-sample)
неверифицирована.

Схема: для каждого фолда (random-0..5, scaffold) fit = 4/5 train, valid = 1/5.
  SENTINEL_DOCS / PAIR_TRUSTED калибруются ТОЛЬКО по fit (calibrate_* в
  lit_si_rule); article_si.csv и paper_si.csv — внешние таблицы статей, их
  построение не использует y (применяются как есть).
Оценка: базовые OOF-предсказания (стекинг-прокси) + оверрайды → MSE на valid.
Варианты: (a) без оверрайдов, (b) article+paper, (c) article+paper+lit(fit).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.lit_si_rule import (calibrate_pair_trusted,
                                     calibrate_sentinel_docs, override_for)
from experiments.stacking import fold_arrays
from experiments.stacking_eval import load_sources
from vacv.article_lookup import article_frame
from vacv.dataset import load
from vacv.literature import attach
from experiments.active_value2 import attach_nn
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds


def valid_lit_overrides(records: dict, frame_valid: pd.DataFrame,
                        sdocs: set[str], ptr: set[str], covered: set[str]):
    """Кандидат-правило (откалиброванное по fit) применяется к valid-строкам."""
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey
    rows = {}
    for idx, row in enumerate(frame_valid.itertuples()):
        mol = Chem.MolFromSmiles(row.SMILES)
        key = MolToInchiKey(mol) if mol else ""
        hit = override_for(records, key, pair_trusted=ptr, sentinel_docs=sdocs)
        if hit is None:
            continue
        sentinel, value, doc, how = hit
        if row.SMILES in covered:
            continue
        rows[row.SMILES] = {"sentinel": sentinel, "si": value}
    return rows


def apply_overrides(fr: pd.DataFrame, art_idx: pd.DataFrame, lit: dict) -> pd.DataFrame:
    out = fr.copy()
    sent = out["SMILES"].map(art_idx["sentinel"]).fillna(False).astype(bool)
    si = pd.to_numeric(out["SMILES"].map(art_idx["si"]), errors="coerce")
    if lit:
        lit_idx = pd.DataFrame([{"smiles": s, **v} for s, v in lit.items()]).set_index("smiles")
        lit_sent = out["SMILES"].map(lit_idx["sentinel"]).fillna(False).astype(bool)
        lit_si = pd.to_numeric(out["SMILES"].map(lit_idx["si"]), errors="coerce")
        sent = sent | lit_sent
        si = si.where(si.notna(), lit_si)
    out["article_sentinel"] = sent
    out["article_si"] = si
    return out


def piece_mse(piece, art: pd.DataFrame) -> float:
    """Прогноз прокси (как в article_override_eval) + оверрайды из piece['frame']."""
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
    return float(np.mean((pred - piece["y"]) ** 2))


def run_split(name: str, src: Path, splits: dict, frame: pd.DataFrame,
              y: np.ndarray, art_all: pd.DataFrame, art_idx: pd.DataFrame,
              covered: set[str], records: dict, rows: list) -> None:
    probs, group = load_sources(src, name)
    pieces = fold_arrays(probs, group, frame, y, splits[name])
    for piece in pieces:
        valid = piece["valid"]
        fit_mask = np.ones(len(y), dtype=bool)
        fit_mask[valid] = False
        # (a) база: без оверрайдов
        fr = frame.iloc[valid].copy()
        fr["article_sentinel"] = False
        fr["article_si"] = np.nan
        piece["frame"] = fr
        base = piece_mse(piece, None)
        # (b) article+paper (внешние таблицы, без y в построении)
        piece["frame"] = apply_overrides(frame.iloc[valid], art_idx, {})
        ap = piece_mse(piece, None)
        # (c) + lit: SENTINEL_DOCS/PAIR_TRUSTED калибруются только по fit,
        #     кандидат-правило применяется к valid-строкам
        sdocs = calibrate_sentinel_docs(records, frame.iloc[fit_mask].reset_index(drop=True),
                                        y[fit_mask])
        ptr = calibrate_pair_trusted(records, frame.iloc[fit_mask].reset_index(drop=True),
                                     y[fit_mask])
        lit = valid_lit_overrides(records, frame.iloc[valid], sdocs, ptr, covered)
        piece["frame"] = apply_overrides(frame.iloc[valid], art_idx, lit)
        full = piece_mse(piece, None)
        rows.append({"split": name, "fold": len(rows) % 5, "base": base, "art+paper": ap,
                     "+lit(fit)": full, "n_lit": len(lit), "n_sdocs": len(sdocs)})
        print(f"{name} fold{len(rows) % 5}: base {base:,.0f} | art+paper {ap:,.0f} | "
              f"+lit(fit) {full:,.0f} ({(full - base) / base:+.2%}) | lit-строк {len(lit)} "
              f"(sdocs {len(sdocs)})", flush=True)


def main() -> None:
    data = load()
    frame, y = attach_nn(attach(data.train)), np.asarray(data.y, dtype=float)
    frame = frame.reset_index(drop=True)

    art_fixed = pd.read_csv("data/processed/article_si.csv")
    paper = pd.read_csv("data/processed/paper_si.csv")
    bmcl = pd.read_csv("data/processed/bmcl_si.csv")
    rcb = pd.read_csv("data/processed/rcb_si.csv")
    art_all = pd.concat([art_fixed, paper, bmcl, rcb], ignore_index=True)  # (b)
    covered = set(art_all["smiles"])
    art_idx = art_all.set_index("smiles")

    records = json.load(open("outputs/doc-scraper/records-by-key.json"))

    splits = {f"random-{s}": [v for _, v in KFold(5, shuffle=True, random_state=s).split(y)]
              for s in (0, 1, 2, 3, 4, 5)}
    splits["scaffold"] = scaffold_folds(scaffolds_of(frame), n_splits=5)

    rows = []
    for name in ("random-0", "random-1", "random-2"):
        run_split(name, Path("outputs/stacking-screen"), splits, frame, y,
                  art_all, art_idx, covered, records, rows)
    for name in ("random-3", "random-4", "random-5", "scaffold"):
        run_split(name, Path("outputs/stacking-confirm"), splits, frame, y,
                  art_all, art_idx, covered, records, rows)

    res = pd.DataFrame(rows)
    print("\n=== средние по фолдам:")
    print(res[["base", "art+paper", "+lit(fit)"]].mean().apply(lambda v: f"{v:,.0f}"))
    b, ap, fl = res["base"].mean(), res["art+paper"].mean(), res["+lit(fit)"].mean()
    print(f"art+paper vs base: {(ap - b) / b:+.2%}; +lit(fit) vs base: {(fl - b) / b:+.2%}; "
          f"+lit vs art+paper: {(fl - ap) / ap:+.2%}")
    res.to_csv("outputs/doc-scraper/lit-si-cv.csv", index=False)
    print("→ outputs/doc-scraper/lit-si-cv.csv")


if __name__ == "__main__":
    main()
