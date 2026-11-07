"""Задача 6 (диагностика нуля): funnel — сколько строк доходит до сверки значений.

Почему в CV ни одна пара (type, doc) не прошла: считаем по полной train
(ПОСТ-ХОК объяснение, не источник правила) на каждом фильтре row_ok:
есть EXT-VACV ratio-запись → не заблокирована has_full_pair → не заблокирована
production-Ratio → candidate() (Rule A + конфликт) None → осталось; и среди
оставшихся — сколько записей бит-в-бит (0.3%) равны y, а сколько мимо.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.active_value2 import attach_nn
from experiments.lit_si_rule import (RATIO, _is_vacv, _to_um,
                                     calibrate_pair_trusted, candidate)
from experiments.qwen_ratio_types_cv import EXT_RATIO_TYPES, TOL_EXACT, build_row_ctx, ikey
from vacv.dataset import load
from vacv.literature import attach

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen-sentinel"

def main() -> None:
    data = load()
    frame, y = attach_nn(attach(data.train)), np.asarray(data.y, dtype=float)
    frame = frame.reset_index(drop=True)
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))
    ctx = build_row_ctx(records, [frame, test])
    ptr = calibrate_pair_trusted(records, frame, y)

    has = blocked_pair = blocked_ratio = blocked_cand = ok = 0
    val_hits: Counter[str] = Counter()
    val_miss: Counter[str] = Counter()
    miss_examples = []
    for idx, row in enumerate(frame.itertuples()):
        smi = row.SMILES
        c = ctx.get(smi)
        if not c:
            continue
        if not c["ratios"]:
            continue
        has += 1
        if c["has_full_pair"]:
            blocked_pair += 1
            continue
        if c["prod_ratio"]:
            blocked_ratio += 1
            continue
        if candidate(records, c["key"], pair_trusted=ptr) is not None:
            blocked_cand += 1
            continue
        ok += 1
        yy = y[idx]
        for t, d, v in c["ratios"]:
            if abs(v - yy) <= TOL_EXACT * max(v, yy, 1e-9):
                val_hits[f"{t}@{d}"] += 1
            else:
                val_miss[f"{t}@{d}"] += 1
                if len(miss_examples) < 20:
                    miss_examples.append({"type": t, "doc": d, "record": round(v, 4),
                                          "y": round(float(yy), 4),
                                          "ratio": round(v / max(float(yy), 1e-9), 4)})

    lines = [
        "# Задача 6: funnel нулевого результата (пост-хок диагностика)",
        "",
        "train-строки с ≥1 EXT-VACV ratio-записью: **%d**" % has,
        "- заблокировано полной VACV-парой в каком-то доке: %d" % blocked_pair,
        "- заблокировано production-Ratio (Rule A уже решает): %d" % blocked_ratio,
        "- заблокировано candidate() (Rule A неоднозначен/конфликт): %d" % blocked_cand,
        "- дошло до сверки значений: **%d**" % ok,
        "",
        "Сверка записей дошедших строк с y (допуск 0.3%):",
        "",
        "| (type, doc) | бит-в-бит | мимо |",
        "|---|---|---|",
    ]
    for k in sorted(set(val_hits) | set(val_miss)):
        lines.append(f"| {k} | {val_hits.get(k, 0)} | {val_miss.get(k, 0)} |")
    if not (val_hits or val_miss):
        lines.append("| _нет строк, дошедших до сверки_ | 0 | 0 |")
    if miss_examples:
        lines += ["", "Примеры расхождений (запись vs y):", "",
                  "| type | doc | record | y | ratio |", "|---|---|---|---|---|"]
        for e in miss_examples:
            lines.append(f"| {e['type']} | {e['doc']} | {e['record']} | {e['y']} | {e['ratio']} |")
    lines += ["", "Запись: значения пост-хок по полной train (объяснение, не источник правила).", ""]
    (OUT / "ratio_types_funnel.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"has {has} | blocked pair {blocked_pair} | blocked ratio {blocked_ratio} "
          f"| blocked candidate {blocked_cand} | ok {ok}")
    print("бит-в-бит:", dict(val_hits) or "—", "| мимо:", dict(val_miss) or "—")
    print(f"→ {OUT / 'ratio_types_funnel.md'}")


if __name__ == "__main__":
    main()
