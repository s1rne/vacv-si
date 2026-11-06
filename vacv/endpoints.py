"""Внешний корпус как два эндпоинта вместо одного отношения.

SI = CC50 / IC50, и обе половины отношения в открытых данных цензурированы справа:
в корпусе `data/processed/vacv_si.csv` знак «>» стоит у 58% IC50 и у 30% CC50.
Прямая регрессия SI по корпусу живёт только на 367 строках `measured`, где точны обе
половины; 389 строк `undetermined` (оба знака «>») дают ровно SI = 1 и не несут
информации о селективности (docs/dataset.md). Разложение на два эндпоинта снимает
эту потерю: строка `IC50 > 100` — полноценное правоцензурированное наблюдение IC50,
даже когда CC50 из той же строки тоже ограничена снизу.

Шкала здесь — log10 концентрации в микромолях, а не pIC50. В ней знак «>» остаётся
правым цензурированием, то есть кодируется как интервал [y, +inf) без смены знака,
который требуется objective `survival:aft` в XGBoost.

Корпус гетерогенен по тест-системе: SI одного и того же ST-246 отличается в 60 раз
между HFF и Vero (docs/research.md, 1.4). Поэтому линия клеток и штамм входят в модель
признаками, а не усредняются. Словари у корпуса и у выборки соревнования разные
(`cell_line` против `Culture`), и приводятся к общему набору токенов.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .paths import PROCESSED

CORPUS = PROCESSED / "vacv_si.csv"

CENSORED_RELATIONS = (">", ">=")

# Линии клеток, встречающиеся и в корпусе, и в выборке соревнования. Порядок важен:
# HELA проверяется до HEL, иначе «HELA cells» попадёт в HEL.
CELL_LINES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vero", ("VERO",)),
    ("bsc", ("BSC", "CV1", "LLCMK2")),
    ("hela", ("HELA",)),
    ("hel", ("HEL", "LUNGFIBROBLAST")),
    ("hff", ("HFF", "FORESKIN")),
    ("mt4", ("MT4",)),
    ("cem", ("CEM",)),
    ("a549", ("A549",)),
    ("prk", ("PRK",)),
    ("e6sm", ("E6SM",)),
    ("bhk", ("BHK",)),
    ("mdck", ("MDCK",)),
    ("hep2", ("HEP2",)),
)

STRAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wr", ("WR", "WESTERNRESERVE")),
    ("copenhagen", ("COPENHAGEN",)),
    ("lister", ("LISTER", "ELSTREE")),
    ("ihd", ("IHD",)),
    ("nycbh", ("NYCBH",)),
    ("tiantan", ("TIANTAN",)),
)

CELL_COLUMNS = [f"cell_{name}" for name, _ in CELL_LINES]
STRAIN_COLUMNS = [f"strain_{name}" for name, _ in STRAINS]
CONTEXT_COLUMNS = CELL_COLUMNS + STRAIN_COLUMNS


def _tokens(text: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _multi_hot(values: pd.Series, vocabulary) -> pd.DataFrame:
    """Многозначное кодирование: «MT-4|Vero» зажигает два столбца, пропуск — ни один."""
    text = values.map(_tokens)
    columns = {}
    for name, patterns in vocabulary:
        hit = np.zeros(len(text))
        for pattern in patterns:
            hit = np.maximum(hit, text.str.contains(pattern, regex=False).to_numpy(float))
        columns[name] = hit
    return pd.DataFrame(columns, index=values.index)


def context_frame(frame: pd.DataFrame, cell_column: str, strain_column: str) -> pd.DataFrame:
    """Единое представление тест-системы для корпуса и для выборки соревнования."""
    cells = _multi_hot(frame[cell_column], CELL_LINES)
    # «HELA cells» содержит подстроку HEL, поэтому HEL гасится там, где сработала HELA.
    cells["hel"] = np.where(cells["hela"] > 0, 0.0, cells["hel"])
    cells.columns = CELL_COLUMNS
    strains = _multi_hot(frame[strain_column], STRAINS)
    strains.columns = STRAIN_COLUMNS
    return pd.concat([cells, strains], axis=1)


def bounds(values: np.ndarray, relations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Интервальная метка для `survival:aft` в XGBoost.

    Границы задаются в исходной шкале микромолей: objective сам логарифмирует метку,
    поэтому отрицательных значений в границах быть не должно. Точное значение — [y, y];
    «>» — [y, +inf); «<» — [0, y]. Левая цензура редка (три строки корпуса), но кодируется
    честно, а не выбрасывается.
    """
    y = np.asarray(values, dtype=np.float64)
    relation = np.asarray(relations, dtype=object)
    lower = y.copy()
    upper = y.copy()
    right = np.isin(relation, CENSORED_RELATIONS)
    left = np.isin(relation, ("<", "<="))
    upper[right] = np.inf
    lower[left] = 0.0
    return lower, upper


def load_corpus(path=CORPUS) -> pd.DataFrame:
    """Корпус с добавленными метками обоих эндпоинтов и общим представлением контекста."""
    table = pd.read_csv(path)
    table = table[(table["ic50"] > 0) & (table["cc50"] > 0)].reset_index(drop=True)
    for endpoint in ("ic50", "cc50"):
        relation = table[f"{endpoint}_relation"].fillna("=")
        lower, upper = bounds(table[endpoint].to_numpy(), relation.to_numpy())
        table[f"log_{endpoint}"] = np.log10(table[endpoint])
        table[f"{endpoint}_lower"] = lower
        table[f"{endpoint}_upper"] = upper
        table[f"{endpoint}_censored"] = np.isin(relation.to_numpy(object), CENSORED_RELATIONS)
    table["inactive"] = table["ic50_censored"].astype(int)
    table["log_si"] = table["log_cc50"] - table["log_ic50"]
    return pd.concat([table, context_frame(table, "cell_line", "strain")], axis=1)


def competition_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Культура и штамм соревнования в тех же столбцах, что у корпуса.

    Поле `strain` соревнования — свободный текст, в который часто вписана и культура
    («Vaccinia virus (strain Copenhagen). culture Vero»), поэтому линия ищется в обоих
    столбцах, а найденное объединяется.
    """
    joined = frame["Culture"].astype(str) + " " + frame["strain"].astype(str)
    return context_frame(pd.DataFrame({"cell": joined, "strain": frame["strain"]}), "cell", "strain")
