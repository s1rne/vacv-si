"""Задача 7 (обратная сторона): не мешает ли ID50 ∈ IC_TYPES матчеру продa.

experiments/lit_si_rule.py уже считает ID50 как антивирусную запись
(_unc_vacv_ic), то есть VACV-ID50 без цензуры БЛОКИРУЕТ сентинель-оверрайд и
пару Rule-A. Если ID50 — не антивирусная мера (гипотеза задачи 7), блокировка
ошибочна: на valid должны появляться новые ВЕРНЫЕ сентины. Протокол — эталон
experiments/lit_si_cv.py: калибровка (calibrate_sentinel_docs/
calibrate_pair_trusted) по fit=4/5; сравнение прод-IC_TYPES vs IC_TYPES\{ID50}
на valid; ложные/верные новые сентины и MSE. 35 фолдов.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

import experiments.lit_si_rule as rule_mod
from experiments.active_value2 import attach_nn
from experiments.lit_si_cv import apply_overrides, piece_mse, valid_lit_overrides
from experiments.lit_si_rule import calibrate_pair_trusted, calibrate_sentinel_docs
from experiments.stacking import fold_arrays
from experiments.stacking_eval import load_sources
from vacv.dataset import load
from vacv.literature import attach
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds

OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)


def run_split(name, src, splits, frame, y, art_idx, covered, records, rows):
    probs, group = load_sources(src, name)
    pieces = fold_arrays(probs, group, frame, y, splits[name])
    for fold, piece in enumerate(pieces):
        valid = piece["valid"]
        fit_mask = np.ones(len(y), dtype=bool)
        fit_mask[valid] = False
        fit_frame = frame.iloc[fit_mask].reset_index(drop=True)
        valid_frame = frame.iloc[valid]
        sdocs_fit = calibrate_sentinel_docs(records, fit_frame, y[fit_mask])
        ptr_fit = calibrate_pair_trusted(records, fit_frame, y[fit_mask])

        lit_a = valid_lit_overrides(records, valid_frame, sdocs_fit, ptr_fit, covered)
        piece["frame"] = apply_overrides(valid_frame, art_idx, lit_a)
        mse_a = piece_mse(piece, None)

        saved = rule_mod.IC_TYPES
        rule_mod.IC_TYPES = saved - {"ID50"}
        try:
            sdocs_b = calibrate_sentinel_docs(records, fit_frame, y[fit_mask])
            ptr_b = calibrate_pair_trusted(records, fit_frame, y[fit_mask])
            lit_b = valid_lit_overrides(records, valid_frame, sdocs_b, ptr_b, covered)
            piece["frame"] = apply_overrides(valid_frame, art_idx, lit_b)
            mse_b = piece_mse(piece, None)
        finally:
            rule_mod.IC_TYPES = saved

        y_by_smi = dict(zip(valid_frame["SMILES"], piece["y"]))
        new_senti = {s for s, v in lit_b.items()
                     if v["sentinel"] and s not in lit_a}
        new_si = {s for s, v in lit_b.items()
                  if not v["sentinel"] and s not in lit_a}
        true_s = [s for s in new_senti if y_by_smi[s] == 10000.0]
        false_s = [s for s in new_senti if y_by_smi[s] != 10000.0]
        rows.append({"split": name, "fold": fold, "mse_prod": mse_a, "mse_no_id50": mse_b,
                     "n_sent_prod": sum(1 for v in lit_a.values() if v["sentinel"]),
                     "new_sent": len(new_senti), "true_sent": len(true_s),
                     "false_sent": len(false_s), "new_pair": len(new_si)})
        print(f"{name} fold{fold}: prod {mse_a:,.0f} | no-ID50 {mse_b:,.0f} "
              f"| новых сентинов {len(new_senti)} (верных {len(true_s)}, ложных {len(false_s)}) "
              f"| новых пар {len(new_si)}", flush=True)


def main() -> None:
    data = load()
    frame, y = attach_nn(attach(data.train)), np.asarray(data.y, dtype=float)
    frame = frame.reset_index(drop=True)
    art_all = pd.concat([pd.read_csv(ROOT / "data/processed/article_si.csv"),
                         pd.read_csv(ROOT / "data/processed/paper_si.csv")],
                        ignore_index=True)
    covered = set(art_all["smiles"])
    art_idx = art_all.set_index("smiles")
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))
    splits = {f"random-{s}": [v for _, v in KFold(5, shuffle=True, random_state=s).split(y)]
              for s in (0, 1, 2, 3, 4, 5)}
    splits["scaffold"] = scaffold_folds(scaffolds_of(frame), n_splits=5)
    rows: list[dict] = []
    for name in ("random-0", "random-1", "random-2"):
        run_split(name, ROOT / "outputs/stacking-screen", splits, frame, y,
                  art_idx, covered, records, rows)
    for name in ("random-3", "random-4", "random-5", "scaffold"):
        run_split(name, ROOT / "outputs/stacking-confirm", splits, frame, y,
                  art_idx, covered, records, rows)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "id50_matcher_cv.csv", index=False)
    print(f"\nсредние: prod {res['mse_prod'].mean():,.0f} | no-ID50 {res['mse_no_id50'].mean():,.0f} "
          f"({(res['mse_no_id50'].mean() - res['mse_prod'].mean()) / res['mse_prod'].mean():+.3%})")
    print(f"новых сентинов {int(res['new_sent'].sum())} (верных {int(res['true_sent'].sum())}, "
          f"ложных {int(res['false_sent'].sum())}); новых пар {int(res['new_pair'].sum())}")
    print(f"→ {OUT / 'id50_matcher_cv.csv'}")


if __name__ == "__main__":
    main()
