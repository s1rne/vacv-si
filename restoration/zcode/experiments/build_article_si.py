"""Генерация data/processed/article_si.csv из таблиц OA-статей.

Пока покрывает CHEMBL3864138 (BMCL 2016, серия 24a/24b) — единственную статью,
где конвенция проверена на train бит-в-бит: EC50 «>100» → IC50 = NaN →
SI = 10000 (сентинел), измеренный EC50 → SI = MCC/EC50 (граница «>100» → 100).
Проверено на 13 train-строках: 11 сентинелов и 2 точных SI (24a11=41.667,
24b11=58.824).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("outputs/doc-scraper")
DEST = Path("data/processed/article_si.csv")


def main() -> None:
    meas = json.load(open(OUT / "pmc-measurements.json"))
    lookup = pd.read_csv("data/processed/competition_open_data_lookup.csv", low_memory=False)
    train = pd.read_csv("data/raw/kaggle/train.csv")
    test = pd.read_csv("data/raw/kaggle/test.csv")
    # числовые Title («10», «5»…) повторяются в разных статьях с разными SI —
    # дубликаты допустимы; проверяем только 24a/24b-метки этой статьи
    pat = train["Title"].astype(str).str.match(r"^24[ab]\d\d$")
    dup = train[pat & train["Title"].duplicated(keep=False)]
    assert dup.empty, "24a/24b titles should be unique"
    y_src = train.drop_duplicates("Title").set_index("Title")["SI VACV"]

    labels = {}
    for m in meas:
        if m["doc"] == "CHEMBL3864138":
            labels.setdefault(m["label"], {}).setdefault(m["kind"], (m["value"], m["relation"], m["sub"]))

    rows = []
    for frame, split in ((train, "train"), (test, "test")):
        for record in frame.to_dict("records"):
            title = str(record["Title"])
            hit = labels.get(title)
            if not hit:
                continue
            ec = hit.get("ic50")  # в этой статье EC50 против вакцинии
            cc = hit.get("cc50")
            if not ec:
                continue
            sentinel = ec[1] == ">"
            si = None
            if not sentinel and cc:
                si = cc[0] / ec[0]
            rows.append({
                "split": split,
                "Title": title,
                "smiles": record["SMILES"],
                "doc": "CHEMBL3864138",
                "label": title,
                "ec50_um": ec[0],
                "ec50_relation": ec[1],
                "cc50_um": cc[0] if cc else None,
                "cc50_relation": cc[1] if cc else None,
                "si": si,
                "sentinel": sentinel,
                "sub": ec[2],
            })

    df = pd.DataFrame(rows)
    # валидация на train: сентинел против y, SI против y
    y = y_src
    tr = df[df["split"] == "train"]
    y_true = y.reindex(tr["Title"].to_numpy()).to_numpy(dtype=float)
    pred_sentinel = tr["sentinel"].to_numpy(dtype=bool)
    sent_ok = bool((pred_sentinel == (y_true == 10000)).all())
    si_rows = tr[~tr["sentinel"]]
    si_ok = all(abs(row["si"] - y.loc[row["Title"]]) < 1e-3 * max(y.loc[row["Title"]], 1)
                for _, row in si_rows.iterrows())
    print(f"train rows: {len(tr)} (sentinel {int(tr['sentinel'].sum())}), "
          f"sentinel agreement: {sent_ok}, si match: {si_ok}")
    if not (sent_ok and si_ok):
        raise ValueError("article extraction disagrees with train")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEST, index=False)
    print(f"{DEST}: {len(df)} rows (train {int((df['split']=='train').sum())}, "
          f"test {int((df['split']=='test').sum())}, sentinels {int(df['sentinel'].sum())})")


if __name__ == "__main__":
    main()
