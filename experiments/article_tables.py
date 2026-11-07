"""Таблицы OA-статей из PMC: IC50/EC50/CC50/SI по меткам соединений.

JATS-XML статьи разбирается на таблицы; у каждой таблицы определяются колонки
с измерениями по заголовкам (IC50, EC50, CC50, TD50, SI, MCC...) с учётом
colspan/rowspan — заголовок разворачивается в прямоугольную сетку, колонка
наследует текст всех ячеек над собой. Значения парсятся вместе с отношениями
(">100" → 100, цензурировано) — именно так соревнование конвертировало
границы в числа: в train строка 24a11 (EC50 2.4, MCC>100) превратилась в
CC50=100.0 и SI=41.667. Единицы — из текста заголовка колонки.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

XHTML_NS = "{http://www.w3.org/1999/xhtml}"

# [a-z]? поглощает буквы-сноски, приклеенные суперскриптом: "MCCb (μM)", "EC50a (μM)"
KIND_PATTERNS = [
    ("si", re.compile(r"\bSI\b|selectivity", re.I)),
    ("cc50", re.compile(r"\bCC50[a-z]?\b|\bMCC[a-z]?\b|cytotox|cytostatic|TCID50|TD50", re.I)),
    ("ic50", re.compile(r"\bIC50[a-z]?\b|\bEC50[a-z]?\b|\bED50[a-z]?\b|\bID50[a-z]?\b", re.I)),
]

LABEL_PATTERN = re.compile(r"\bCompd|Compound\b|\bNo\.?\b|\bCmpd|Code|Entry", re.I)

NUM_RE = re.compile(r"([<>≥≤~])?\s*([0-9]+(?:\.[0-9]+)?(?:\s*[±×]\s*[0-9]+(?:\.[0-9]+)?)?)")


@dataclass
class Cell:
    text: str = ""
    value: float | None = None
    relation: str = ""


@dataclass
class Table:
    doc: str
    table_id: str
    caption: str
    header_text: list[str]      # склеенный заголовок колонки
    sub_header: list[str]       # нижняя строка заголовка (подгруппа: "Vaccinia virus")
    rows: list[list[Cell]]
    column_kinds: dict[int, str] = field(default_factory=dict)
    units: dict[int, str] = field(default_factory=dict)
    label_col: int = 0


def _text(elem) -> str:
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def _parse_cell(text: str) -> Cell:
    cell = Cell(text=text)
    m = NUM_RE.search(text)
    if m:
        rel, num = m.group(1) or "", m.group(2)
        num = re.sub(r"[±×].*$", "", num).strip()
        try:
            cell.value = float(num)
            cell.relation = rel
        except ValueError:
            pass
    return cell


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _expand(header_trs: list) -> list[list[str]]:
    """Строки th/td с colspan/rowspan → прямоугольная сетка текстов."""
    occupied: dict[tuple[int, int], str] = {}
    width = 0
    for r, tr in enumerate(header_trs):
        c = 0
        for cell in tr:
            if _local(cell.tag) not in ("th", "td"):
                continue
            while (r, c) in occupied:
                c += 1
            span = int(cell.get("colspan", "1") or 1)
            rows = int(cell.get("rowspan", "1") or 1)
            text = _text(cell)
            for rr in range(r, r + rows):
                for cc in range(c, c + span):
                    occupied[(rr, cc)] = text
            c += span
            width = max(width, c)
    height = len(header_trs)
    return [[occupied.get((r, c), "") for c in range(width)] for r in range(height)]


def _body_rows(table, header_trs: list) -> list[list[Cell]]:
    rows = []
    for section in table:
        if _local(section.tag) != "tbody":
            continue
        for tr in section:
            if _local(tr.tag) != "tr":
                continue
            cells: list[Cell] = []
            for cell in tr:
                if _local(cell.tag) not in ("th", "td"):
                    continue
                cells.append(_parse_cell(_text(cell)))
            if cells:
                rows.append(cells)
    if not rows:
        # таблица без tbody: все tr, кроме заголовочных
        for tr in table.iter():
            if _local(tr.tag) == "tr" and tr not in header_trs:
                cells = [_parse_cell(_text(c)) for c in tr if _local(c.tag) in ("th", "td")]
                if cells:
                    rows.append(cells)
    return rows


UNIT_RE = re.compile(r"µM|μM|uM|nM|mM|µg/mL|μg/mL|ug/mL|mg/mL|µg|μg|ug|mg|µl|μl|ul")


def _units_of(text: str) -> str:
    m = UNIT_RE.search(text)
    return m.group(0) if m else ""


def parse_article(doc: str, xml_path: Path) -> list[Table]:
    root = ET.parse(xml_path).getroot()
    tables = []
    for wrap in root.iter():
        if _local(wrap.tag) != "table-wrap":
            continue
        table_elem = next((el for el in wrap.iter() if _local(el.tag) == "table"), None)
        if table_elem is None:
            continue
        caption_el = next((el for el in wrap.iter() if _local(el.tag) == "caption"), None)
        caption = _text(caption_el) if caption_el is not None else ""
        # Заголовок — все строки <thead>; в some статьях ячейки размечены <td>, не <th>.
        thead = next((el for el in table_elem.iter() if _local(el.tag) == "thead"), None)
        if thead is not None:
            header_trs = [tr for tr in thead if _local(tr.tag) == "tr"]
        else:
            header_trs = [tr for tr in table_elem.iter() if _local(tr.tag) == "tr"
                          and any(_local(c.tag) == "th" for c in tr)]
        grid = _expand(header_trs) if header_trs else []
        rows = _body_rows(table_elem, header_trs)
        if not rows or not grid:
            continue
        n_cols = max(max((c.text and 1 or 0) for c in row) or len(row) for row in rows)
        n_cols = max(n_cols, len(grid[0]) if grid else 0)
        header_text, sub_header = [], []
        for col in range(n_cols):
            texts = [grid[r][col] for r in range(len(grid)) if col < len(grid[r]) and grid[r][col]]
            header_text.append(" ".join(dict.fromkeys(texts)))
            sub_header.append(texts[-1] if texts else "")
        table = Table(doc=doc, table_id=wrap.get("id") or "", caption=caption[:300],
                      header_text=header_text, sub_header=sub_header, rows=rows)
        for col, text in enumerate(header_text):
            for kind, pattern in KIND_PATTERNS:
                if pattern.search(text):
                    table.column_kinds[col] = kind
                    table.units[col] = _units_of(text)
                    break
        for col, text in enumerate(header_text):
            if LABEL_PATTERN.search(text):
                table.label_col = col
                break
        tables.append(table)
    return tables


def measurement_rows(tables: list[Table]) -> list[dict]:
    """Плоский список измерений: {doc, table, caption, label, col, kind, value, relation, unit, sub}."""
    out = []
    for t in tables:
        for row in t.rows:
            label = row[t.label_col].text if t.label_col < len(row) else ""
            if not label:
                continue
            for col, kind in t.column_kinds.items():
                if col >= len(row):
                    continue
                cell = row[col]
                if cell.value is None:
                    continue
                out.append({
                    "doc": t.doc, "table": t.table_id, "caption": t.caption,
                    "label": label, "col": col, "kind": kind,
                    "value": cell.value, "relation": cell.relation,
                    "unit": t.units.get(col, ""), "sub": t.sub_header[col],
                })
    return out
