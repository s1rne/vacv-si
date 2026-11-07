"""Валидация прямых VACV SI-записей из ChEMBL (qwen-находка, задача 3).

records-by-key.json содержит типы «Ratio CC50/IC50», «Selectivity Index» и др. —
готовые значения SI, депонированные авторами статей. Кандидаты для хард-оверрайдов,
но сначала строгая валидация на train бит-в-бит (допуск 0.3%, как в paper_si_match).

Правила отбора для test-оверрайдов (выводятся из train-диагностики):
  - тип из SI_TYPES, assay ~ 'vaccin', value парсится, единицы None;
  - несцензурированные; цензурированные проверяем отдельно (что с ними делает y);
  - конфликт значений (>0.5% между записями) → строка пропускается;
  - строки, уже покрытые article_si/paper_si, не трогаем (там своя валидация).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "doc-scraper"
SI_TYPES = {"Ratio CC50/IC50", "Selectivity Index", "Selectivity index", "SI",
            "Ratio CC50/EC50", "Therapeutic index", "TI"}


def vacv_si_records(records: dict, key: str):
    """[(value, relation, doc, type, assay)] — только VACV-ассеи, парсимое значение."""
    out = []
    for r in records.get(key, []):
        if r.get("type") not in SI_TYPES:
            continue
        if "vaccin" not in str(r.get("assay") or "").lower():
            continue
        try:
            v = float(r.get("value"))
        except (TypeError, ValueError):
            continue
        if not (v > 0):
            continue
        out.append((v, (r.get("relation") or "").strip(), r.get("doc"),
                    r["type"], str(r.get("assay"))[:90]))
    return out


def main() -> None:
    records = json.load(open(OUT / "records-by-key.json"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchiKey
    RDLogger.DisableLog("rdApp.*")

    def ikey(s):
        m = Chem.MolFromSmiles(s)
        return MolToInchiKey(m) if m else None

    diag = []
    for split, path in (("train", "data/raw/kaggle/train.csv"),
                        ("test", "data/raw/kaggle/test.csv")):
        frame = pd.read_csv(path)
        frame["key"] = frame["SMILES"].map(ikey)
        for idx, row in frame.iterrows():
            recs = vacv_si_records(records, row["key"])
            if not recs:
                continue
            vals = sorted({round(v, 6) for v, rel, *_ in recs})
            rels = {rel for _, rel, *_ in recs}
            item = {"split": split, "index": int(idx), "Title": str(row["Title"]),
                    "n_recs": len(recs), "vals": vals, "censored": sorted(rels),
                    "docs": sorted({d for _, _, d, _, _ in recs}),
                    "types": sorted({t for *_, t, _ in recs})}
            if split == "train":
                y = float(row["SI VACV"])
                item["y"] = y
                item["exact"] = any(abs(v - y) <= 0.003 * max(v, y) for v in vals)
            diag.append(item)

    diag_path = OUT / "literature-si-diag.json"
    json.dump(diag, open(diag_path, "w"), indent=1)

    tr = [d for d in diag if d["split"] == "train"]
    exact = [d for d in tr if d.get("exact")]
    print(f"train: строк с VACV SI-записями {len(tr)}, бит-в-бит {len(exact)}")
    for d in tr:
        mark = "OK " if d.get("exact") else "   "
        y = d.get("y")
        y_txt = f"y={y:.2f}" if y is not None else ""
        print(f"  {mark}#{d['index']:>3} {d['Title']:<6} vals={d['vals']} cens={d['censored']} "
              f"{y_txt} docs={d['docs']} types={d['types']}")
    print()
    print(f"test: строк с VACV SI-записями {len([d for d in diag if d['split'] == 'test'])}")
    multi = [d for d in diag if d["split"] == "test" and len(d["vals"]) > 1]
    print(f"  test с несколькими разными значениями: {len(multi)}")
    n_recs_c = Counter(d["n_recs"] for d in diag if d["split"] == "test")
    print(f"  test по числу записей: {dict(sorted(n_recs_c.items()))}")
    print(f"→ {diag_path}")


if __name__ == "__main__":
    main()
