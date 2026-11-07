"""Конвертация paper-si-match.json → data/processed/paper_si.csv.

Формат как у article_si.csv (Title, sentinel, si, cc50_um, smiles) — файл
подхватывается vacv.article_lookup.article_frame() вместе с 24a/24b.
Ключ (smiles, Title): один SMILES может быть покрыт из разных строк
соревнования (test '11'/'15a'), значения разводятся по Title в
vacv.article_lookup.override_lookup.
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
        sentinel = m["rule"] in ("sentinel", "label_sentinel", "sentinel_yfix")
        si = None if sentinel else float(m["si"])
        # ключ (smiles, Title): один SMILES может нести разные значения из
        # разных строк соревнования (test '11' PMC9533879 vs '15a' PMC9865139)
        k = (smiles, title)
        prev = rows.get(k)
        if prev is not None and (prev["sentinel"] != sentinel or
                                 (not sentinel and abs(prev["si"] - si) > 1e-6)):
            raise ValueError(f"конфликт оверрайдов для {smiles}/{title}: {prev} vs {si}")
        rows[k] = {
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
