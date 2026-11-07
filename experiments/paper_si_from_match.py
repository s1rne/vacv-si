"""Конвертация paper-si-match.json → data/processed/paper_si.csv.

Формат как у article_si.csv (Title, sentinel, si, cc50_um, smiles) — файл
подхватывается vacv.article_lookup.article_frame() вместе с 24a/24b.
Матч по SMILES безопасен: коллизии проверены (одинаковые SMILES несут
одинаковый SI, конфликтов с train нет).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("outputs/doc-scraper")


def main() -> None:
    matches = json.load(open(OUT / "paper-si-match.json"))
    test = pd.read_csv("data/raw/kaggle/test.csv")
    train = pd.read_csv("data/raw/kaggle/train.csv")

    rows = {}
    for m in matches:
        frame = test if m["split"] == "test" else train
        idx = m["index"]
        smiles = frame.loc[idx, "SMILES"]
        title = str(frame.loc[idx, "Title"])
        sentinel = m["rule"] in ("sentinel", "label_sentinel")
        si = None if sentinel else float(m["si"])
        prev = rows.get(smiles)
        if prev is not None and (prev["sentinel"] != sentinel or
                                 (not sentinel and abs(prev["si"] - si) > 1e-6)):
            raise ValueError(f"конфликт оверрайдов для {smiles}: {prev} vs {title}/{si}")
        rows[smiles] = {
            "Title": title, "sentinel": sentinel, "si": si, "smiles": smiles,
            "pmc": m["pmc"], "label": m["label"], "rule": m["rule"],
        }
    out = pd.DataFrame(rows.values())
    path = Path("data/processed/paper_si.csv")
    out.to_csv(path, index=False)
    n_sent = int(out["sentinel"].sum())
    print(f"{len(out)} оверрайдов ({n_sent} сентинелов) → {path}")
    print(out.groupby("pmc").size())


if __name__ == "__main__":
    main()
