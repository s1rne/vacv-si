"""Таблицы CC50/IC50/SI из OA-статей (PMC HTML) + матчинг к строкам соревнования.

Лейблы kaggle НЕ совпадают с лейблами статей (сдвинуты: наш train 4c = статья 5c
и т.п.), поэтому матчим по измерениям: (CC50, IC50) из ChEMBL-записей нашей
строки против (CC50, IC50) строки таблицы. Конвенция SI валидирована на train
бит-в-бит: y = CC50/IC50 с 3 знаками; IC50 censored (NA / '>x') → сентинел
10000 даже при измеренном CC50; граница CC50 '>100' → 100 (MCC).

Выход: outputs/doc-scraper/paper-si-rows.json
  [{pmc, label, cc50_um, ic50_um, ic50_censored, cc50_censored, si_table}]
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path("outputs/doc-scraper")
CLEAN = lambda s: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()


def _num(token: str) -> float | None:
    """14.5 ± 4.4 / >309.3 / ∼214.6 / 149 175 → число до ±."""
    token = token.replace("&lt;", "<").replace("&gt;", ">")
    token = token.split("±")[0].strip()
    token = re.sub(r"[><~≈na /]+", "", token.lower().replace("n/a", ""))
    token = token.replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


def parse_vacv_rows(pmc: str, html_path: Path, table_index: int, columns: dict) -> list[dict]:
    """columns: индексы колонок {'label','cc50','ic50','si'}."""
    html = html_path.read_text()
    tables = re.findall(r"<table[^>]*>.*?</table>", html, re.S)
    rows_out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tables[table_index], re.S):
        cells = [CLEAN(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if len(cells) < max(columns.values()) + 1:
            continue
        label = cells[columns["label"]]
        if not label or label.lower().startswith(("compound", "no.")):
            continue
        cc_raw = cells[columns["cc50"]]
        ic_raw = cells[columns["ic50"]]
        cc = _num(cc_raw)
        ic = _num(ic_raw)
        ic_cens = ("<" in ic_raw or ">" in ic_raw or "na" in ic_raw.lower() or "n/a" in ic_raw.lower())
        cc_cens = ">" in cc_raw
        si_raw = cells[columns["si"]] if "si" in columns else ""
        si = _num(si_raw)
        rows_out.append(
            {
                "pmc": pmc,
                "label": label,
                "cc50_um": cc,
                "ic50_um": ic,
                "ic50_censored": ic_cens,
                "cc50_censored": cc_cens,
                "si_table": si,
                "raw": {"cc50": cc_raw, "ic50": ic_raw, "si": si_raw},
            }
        )
    return rows_out


def parse_ejmc_rows(pmc: str, html_path: Path) -> list[dict]:
    """EJMC 2021 (PMC9533879): Compounds | CC50 | IC50/SI (VACV) | IC50/SI (CPXV) | ...

    Ячейка IC50-колонки вида '0.133/>2325' = IC50/SI; 'n/a'/'n/t' = не измерено.
    """
    html = html_path.read_text()
    tables = re.findall(r"<table[^>]*>.*?</table>", html, re.S)
    rows_out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tables[0], re.S):
        cells = [CLEAN(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if len(cells) < 3 or cells[0].lower().startswith("compound"):
            continue
        label = cells[0]
        cc_raw = cells[1].replace("~", "").replace("∼", "")
        vacv = cells[2]
        if "/" not in vacv:
            continue  # строка без IC50/SI пары
        ic_raw, si_raw = vacv.split("/", 1)
        cc = _num(cc_raw)
        ic = _num(ic_raw)
        ic_cens = ic is None or "<" in ic_raw or ">" in ic_raw
        si = _num(si_raw)
        rows_out.append(
            {
                "pmc": pmc,
                "label": label,
                "cc50_um": cc,
                "ic50_um": None if ic_cens else ic,
                "ic50_censored": ic_cens,
                "cc50_censored": ">" in cc_raw,
                "si_table": si,
                "raw": {"cc50": cc_raw, "ic50": ic_raw, "si": si_raw},
            }
        )
    return rows_out


def _expand_grid(table_html: str) -> list[list[str]]:
    """Разворачивает colspan/rowspan в прямоугольную сетку текстов."""
    rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S)
    grid: list[list[str]] = []
    occupied: set = set()
    for ri, row_html in enumerate(rows_html):
        while len(grid) <= ri:
            grid.append([])
        row = grid[ri]
        cells = re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row_html, re.S)
        col = 0
        for attrs, body in cells:
            while (ri, col) in occupied:
                col += 1
            m = re.search(r'colspan="?(\d+)', attrs)
            colspan = int(m.group(1)) if m else 1
            m = re.search(r'rowspan="?(\d+)', attrs)
            rowspan = int(m.group(1)) if m else 1
            text = CLEAN(body)
            for c in range(colspan):
                for r_ in range(rowspan):
                    if r_ > 0:
                        occupied.add((ri + r_, col + c))
                while len(grid) <= ri or len(grid[ri]) <= col + c:
                    while len(grid) <= ri:
                        grid.append([])
                    grid[ri].append("")
                if r_ == 0 or text:
                    grid[ri + r_][col + c] = text
            col += colspan
    return grid


def parse_cage_amide_rows(pmc: str, html_path: Path) -> list[dict]:
    """Viruses 2022 (PMC9865139): Compound (структуры-картинки) | TC50 µM | IC50 µM | SI.

    Лейблы в таблице — изображения, поэтому восстанавливаю их по порядку из
    подписи таблицы «SI values for amides 12a,b–18a,b and bromantane» и
    текста статьи («SI = 18,214 for the leader compound 18a» — 12-я строка
    данных с SI 18,214; значит порядок 12a,12b,13a,13b,14a,14b,15a,15b,
    16a,16b,17a,17b,18a,18b,bromantane,cidofovir).
    """
    labels = ["12a", "12b", "13a", "13b", "14a", "14b", "15a", "15b",
              "16a", "16b", "17a", "17b", "18a", "18b", "bromantane", "cidofovir"]
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    tables = re.findall(r"<table[^>]*>.*?</table>", html, re.S)
    grid = _expand_grid(tables[0])
    rows_out = []
    data_rows = []
    for row in grid:
        if len(row) < 4:
            continue
        cc_raw = row[1]
        if not cc_raw or "TC50" in cc_raw:
            continue
        data_rows.append(row)
    for i, row in enumerate(data_rows):
        cc_raw, ic_raw, si_raw = row[1], row[2], row[3]
        label = labels[i] if i < len(labels) else f"row{i}"
        cc = _num(cc_raw)
        ic = _num(ic_raw)
        ic_cens = "na" in ic_raw.lower() or ">" in ic_raw or "<" in ic_raw or not ic_raw.strip()
        rows_out.append({
            "pmc": pmc, "label": label,
            "cc50_um": cc, "ic50_um": None if ic_cens else ic,
            "ic50_censored": ic_cens, "cc50_censored": ">" in cc_raw,
            "si_table": _num(si_raw),
            "raw": {"cc50": cc_raw, "ic50": ic_raw, "si": si_raw},
        })
    return rows_out


def main() -> None:
    rows: list[dict] = []
    # PMC11331333 (D4MD00181H, RSC Med Chem 2024): No. | R1 | R2 | CC50 | IC50 VACV | SI
    rows += parse_vacv_rows(
        "PMC11331333", OUT / "pmc2/PMC11331333.html", 4,
        {"label": 0, "cc50": 3, "ic50": 4, "si": 5},
    )
    # PMC6194496 (C8MD00347E, MedChemComm 2018): Compound | TC50 | IC50 | SI
    rows += parse_vacv_rows(
        "PMC6194496", OUT / "pmc2/PMC6194496.html", 0,
        {"label": 0, "cc50": 1, "ic50": 2, "si": 3},
    )
    # PMC7651861 (D0MD00108B, RSC Med Chem 2020): Compound | CC50 | IC50 | SI
    rows += parse_vacv_rows(
        "PMC7651861", OUT / "pmc2/PMC7651861.html", 0,
        {"label": 0, "cc50": 1, "ic50": 2, "si": 3},
    )
    rows += parse_ejmc_rows("PMC9533879", OUT / "pmc2/PMC9533879.html")
    rows += parse_cage_amide_rows("PMC9865139", OUT / "pmc2/PMC9865139.html")
    path = OUT / "paper-si-rows.json"
    path.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} строк из 3 статей → {path}")
    for r in rows:
        print(f"  {r['pmc']} {r['label']:<10} CC50={r['cc50_um']} IC50={r['ic50_um']} "
              f"ic_cens={r['ic50_censored']} SI={r['si_table']}")


if __name__ == "__main__":
    main()
