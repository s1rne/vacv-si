"""Страж: запрещённые колонки не читаются нигде, кроме диагностики.

Правила соревнования запрещают использовать CC50_uM и IC50_uM_VACV для построения
модели и предсказания. Единственное исключение — vacv/exact_si.py: он восстанавливает
истинную метку теста, чтобы измерить собственную ошибку, и намеренно не участвует ни
в обучении, ни в выборе модели.
"""

from __future__ import annotations

import pathlib

FORBIDDEN = ("CC50_uM", "IC50_uM_VACV")
ALLOWED = {"vacv/exact_si.py", "vacv/dataset.py", "vacv/datacard.py", "tests/test_forbidden_columns.py"}
ROOTS = ("vacv", "scripts", "experiments")


def source_files():
    root = pathlib.Path(__file__).resolve().parent.parent
    for folder in ROOTS:
        for path in (root / folder).rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            if relative not in ALLOWED:
                yield relative, path.read_text(encoding="utf-8")


def code_lines(text: str):
    """Строки кода без комментариев и без строк документации."""
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.count('"""') % 2:
            inside = not inside
            continue
        if inside or stripped.startswith("#"):
            continue
        yield line.split("#", 1)[0]


def test_no_module_reads_the_forbidden_columns():
    offenders = {
        name: [line.strip() for line in code_lines(text) if any(column in line for column in FORBIDDEN)]
        for name, text in source_files()
    }
    offenders = {name: lines for name, lines in offenders.items() if lines}
    assert not offenders, f"запрещённые колонки читаются в: {offenders}"
