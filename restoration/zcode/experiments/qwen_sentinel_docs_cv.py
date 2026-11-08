"""Задача 5 (fold-internal, по требованию управляющего агента).

Эталон — experiments/lit_si_cv.py: правила (SENTINEL_DOCS/PAIR_TRUSTED)
каллибруются ТОЛЬКО по fit=4/5 (calibrate_sentinel_docs/calibrate_pair_trusted),
кандидат-правило применяется к valid=1/5; оценка — сохранённые OOF-прокси
(outputs/stacking-screen|confirm) через тот же postprocess (piece_mse).

Проверяемые варианты на каждом фолде:
  (c) prod          — калибровка по fit (текущее производство);
  (d) prod+forced   — плюс ручной кандидат CHEMBL4368876 в SENTINEL_DOCS
                      (док CC50 '>250000 nM' для test 6#447/7#240): смотрим,
                      сколько ложных сентинов он даёт на valid и MSE-цену;
  (e) fixed-const   — наоборот, захардкоженный SENTINEL_DOCS без fit-калибровки
                      (мера того, что fit-калибровка добавляет на valid).
Дополнительно логируется: в каких фолдах fit-калибровка находит новые доки
(CHEMBL3864138 — камфорный/урацильный 24-серийный набор) и никогда ли —
CHEMBL4368876.

Запрещённые колонки не используются (y = разрешённая SI VACV из train).

Выход: outputs/qwen-sentinel/sentinel_docs_cv.csv (+ печать по фолдам).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.active_value2 import attach_nn
from experiments.lit_si_cv import apply_overrides, piece_mse, valid_lit_overrides
from experiments.lit_si_rule import (SENTINEL_DOCS, calibrate_pair_trusted,
                                     calibrate_sentinel_docs)
from experiments.stacking import fold_arrays
from experiments.stacking_eval import load_sources
from vacv.dataset import load
from vacv.literature import attach
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)

#: ручной кандидат-док из форензики (test 6#447/7#240, CC50 '>250000 nM' Vero)
FORCED = {"CHEMBL4368876"}


def run_split(name: str, src: Path, splits: dict, frame: pd.DataFrame,
              y: np.ndarray, art_idx: pd.DataFrame, covered: set[str],
              records: dict, rows: list) -> None:
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

        # (c) prod: всё калибруется по fit
        lit_c = valid_lit_overrides(records, valid_frame, sdocs_fit, ptr_fit, covered)
        piece["frame"] = apply_overrides(valid_frame, art_idx, lit_c)
        mse_c = piece_mse(piece, None)

        # (d) prod + ручной док
        lit_d = valid_lit_overrides(records, valid_frame, sdocs_fit | FORCED, ptr_fit, covered)
        piece["frame"] = apply_overrides(valid_frame, art_idx, lit_d)
        mse_d = piece_mse(piece, None)

        # (e) фиксированные константы прод-SENTINEL_DOCS (без fit-калибровки)
        lit_e = valid_lit_overrides(records, valid_frame, set(SENTINEL_DOCS), ptr_fit, covered)
        piece["frame"] = apply_overrides(valid_frame, art_idx, lit_e)
        mse_e = piece_mse(piece, None)

        y_by_smi = dict(zip(valid_frame["SMILES"], piece["y"]))
        sent_c = {s for s, v in lit_c.items() if v["sentinel"]}
        sent_d = {s for s, v in lit_d.items() if v["sentinel"]}
        new_forced = sorted(sent_d - sent_c)
        false_new = [s for s in new_forced if y_by_smi[s] != 10000.0]
        false_c = [s for s, v in lit_c.items() if v["sentinel"] and y_by_smi[s] != 10000.0]

        rows.append({
            "split": name, "fold": fold,
            "mse_prod_fit": mse_c, "mse_forced": mse_d, "mse_fixed": mse_e,
            "n_sdocs_fit": len(sdocs_fit),
            "fit_docs_beyond_const": ",".join(sorted(sdocs_fit - set(SENTINEL_DOCS))) or "—",
            "const_docs_kept": len(sdocs_fit & set(SENTINEL_DOCS)),
            "4368876_in_fit": "CHEMBL4368876" in sdocs_fit,
            "n_sent_prod": len(sent_c), "n_sent_forced": len(sent_d),
            "forced_new_valid_sent": len(new_forced),
            "forced_false": len(false_new), "prod_false": len(false_c),
            "n_lit_prod": len(lit_c),
        })
        print(f"{name} fold{fold}: prod {mse_c:,.0f} | forced {mse_d:,.0f} | "
              f"fixed {mse_e:,.0f} | sdocs_fit={len(sdocs_fit)} "
              f"(новые: {sorted(sdocs_fit - set(SENTINEL_DOCS)) or '—'}) "
              f"| прод-сентинов {len(sent_c)} (ложных {len(false_c)}) "
              f"| forced-новых {len(new_forced)} (ложных {len(false_new)})", flush=True)


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
    res.to_csv(OUT / "sentinel_docs_cv.csv", index=False)
    print("\n=== средние по фолдам:")
    print(res[["mse_prod_fit", "mse_forced", "mse_fixed"]].mean()
          .apply(lambda v: f"{v:,.0f}"))
    print(f"forced vs prod: {(res['mse_forced'].mean() - res['mse_prod_fit'].mean()) / res['mse_prod_fit'].mean():+.3%}")
    print(f"fixed vs prod : {(res['mse_fixed'].mean() - res['mse_prod_fit'].mean()) / res['mse_prod_fit'].mean():+.3%}")
    print(f"фолдов, где fit-калибровка нашла CHEMBL3864138: "
          f"{int(res['fit_docs_beyond_const'].str.contains('CHEMBL3864138').sum())}/{len(res)}")
    print(f"фолдов, где fit-калибровка нашла CHEMBL4368876: "
          f"{int(res['4368876_in_fit'].sum())}/{len(res)}")
    print(f"всего ложных прод-сентинов на valid: {int(res['prod_false'].sum())}; "
          f"ложных от форсированного дока: {int(res['forced_false'].sum())}")
    print(f"всего valid-срабатываний форсированного дока: {int(res['forced_new_valid_sent'].sum())}")
    print(f"→ {OUT / 'sentinel_docs_cv.csv'}")


if __name__ == "__main__":
    main()
