"""Извлечение SI для ic50-only соединений из таблиц их статей-источников.

Матчинг строки таблицы с нашей строкой данных (в порядке надёжности):
1. метка: Title из competition_open_data_lookup.csv == label таблицы —
   соединение точно из этой статьи, CC50 берём из его собственной строки;
2. значение: best_ic50_um (медиана записей ChEMBL) ≈ EC50/IC50 строки.
   Если в таблице несколько IC50-колонок, используется только та, у которой
   в подзаголовке «Vaccinia»; допуск 2%.

Из сматченной строки берётся CC50 (или SI напрямую, если колонка есть).
Границы конвертируются так же, как это делало соревнование: «>100» → 100
(проверено на train 24a11: EC50 2.4, MCC>100 → SI=41.667).

Валидация: для train-строк извлечённый SI сравнивается с истинным y.
Выход: outputs/doc-scraper/cc50-recovery.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/doc-scraper")
VALUE_TOL = 0.02


def unit_scale(unit: str) -> float | None:
    return {"": 1.0, "µM": 1.0, "μM": 1.0, "uM": 1.0, "nM": 1e-3, "mM": 1e3}.get(unit)


def main() -> None:
    docs = json.load(open("outputs/ic50_only_docs.json"))
    lookup = pd.read_csv("data/processed/competition_open_data_lookup.csv", low_memory=False)
    presence = pd.read_csv("data/processed/competition_presence.csv", low_memory=False)

    lk = lookup[lookup["smiles"].isin(docs.keys())].set_index("smiles")
    pr = presence[presence["smiles"].isin(docs.keys())].set_index("smiles")

    measurements = json.load(open(OUT / "pmc-measurements.json"))
    # группировка по (doc, table, label): одна строка бумаги = один кандидат
    rows_by_table: dict[tuple, list[dict]] = {}
    for m in measurements:
        key = (m["doc"], m["table"], m["label"])
        rows_by_table.setdefault(key, []).append(m)

    results = []
    for smiles, doc_list in docs.items():
        if smiles not in lk.index:
            continue
        title = str(lk.loc[smiles, "Title"]).strip()
        ic50 = pr.loc[smiles, "best_ic50_um"]
        ic50 = float(ic50) if pd.notna(ic50) else None
        candidates = []
        for (doc, table, label), cells in rows_by_table.items():
            if doc not in doc_list:
                continue
            by_label = label.strip().lower() == title.lower()
            ic50_cells = [c for c in cells if c["kind"] == "ic50" and unit_scale(c["unit"])]
            vacc = [c for c in ic50_cells if "vaccin" in c["sub"].lower()]
            ic50_cell = (vacc or ic50_cells or [None])[0]
            by_value = False
            if ic50 is not None and ic50_cell is not None:
                row_ic50 = ic50_cell["value"] * unit_scale(ic50_cell["unit"])
                by_value = abs(row_ic50 - ic50) <= VALUE_TOL * max(row_ic50, ic50)
            if not (by_label or by_value):
                continue
            cc50_cell = next((c for c in cells if c["kind"] == "cc50" and unit_scale(c["unit"])), None)
            si_cell = next((c for c in cells if c["kind"] == "si" and unit_scale(c["unit"])), None)
            candidates.append({
                "smiles": smiles, "title": title, "doc": doc, "table": table, "label": label,
                "match": ("label+value" if by_label and by_value else
                          "label" if by_label else "value"),
                "ic50_um": ic50,
                "row_ic50": (ic50_cell["value"] * unit_scale(ic50_cell["unit"])) if ic50_cell else None,
                "row_ic50_rel": ic50_cell["relation"] if ic50_cell else None,
                "row_ic50_sub": ic50_cell["sub"] if ic50_cell else None,
                "cc50": (cc50_cell["value"] * unit_scale(cc50_cell["unit"])) if cc50_cell else None,
                "cc50_rel": cc50_cell["relation"] if cc50_cell else None,
                "si": (si_cell["value"] * unit_scale(si_cell["unit"])) if si_cell else None,
                "si_rel": si_cell["relation"] if si_cell else None,
            })
        # лучший кандидат: label+value > label > value; при прочих равных — с CC50
        candidates.sort(key=lambda c: (
            {"label+value": 0, "label": 1, "value": 2}[c["match"]],
            c["cc50"] is None,
        ))
        if candidates:
            results.append(candidates[0])

    for r in results:
        if r["si"] is not None:
            r["si_est"] = r["si"]
        elif r["cc50"] is not None and r["row_ic50"]:
            r["si_est"] = r["cc50"] / r["row_ic50"]
        elif r["cc50"] is not None and r["ic50_um"]:
            r["si_est"] = r["cc50"] / r["ic50_um"]
        else:
            r["si_est"] = None

    (OUT / "cc50-recovery.json").write_text(json.dumps(results, indent=1))
    n = len(results)
    with_cc = sum(1 for r in results if r["cc50"] is not None)
    with_si = sum(1 for r in results if r["si_est"] is not None)
    print(f"matched rows: {n}, with CC50: {with_cc}, with SI estimate: {with_si}")
    for r in results:
        flag = "" if r["cc50"] else "  (no cc50)"
        print(f"  {r['title'][:16]:16} {r['match']:11} ic50={r['ic50_um']} "
              f"row={r['row_ic50'] if r['row_ic50'] is None else round(r['row_ic50'], 3)} "
              f"cc50={r['cc50']} si_est={round(r['si_est'], 3) if r['si_est'] else None}{flag}")


if __name__ == "__main__":
    main()
