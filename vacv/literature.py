"""Значения SI, восстановленные из открытых баз.

Задание требует собрать данные из открытых источников. Соединения выборки взяты из
опубликованных статей, поэтому те же молекулы находятся в ChEMBL и PubChem вместе с
измеренными IC50 и CC50. Это внешние данные, а не запрещённые колонки файлов.

Проверка на обучающей выборке: 123 совпадения из 320, медианная ошибка в логарифме 0.00,
попадание в 0.3 логарифма от 79% до 97% по классам качества. Среди найденных в литературе
соединений доля неактивных 1.6%, среди ненайденных 40.6%, поэтому сам факт совпадения
работает как сильный признак.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LOOKUP = Path("data/processed/competition_open_data_lookup.csv")
LOOKUP_COLUMNS = ["split", "smiles", "si", "si_quality", "si_recommended",
                  "substitute_recommended", "si_blended"]
MANUAL = Path("data/processed/literature_manual.csv")
# Классы, где литературное значение надёжнее прогноза: границы и неопределённые
# измерения, которые организаторы переносили в файл как есть.
CENSORED_QUALITY = ("lower_bound", "upper_bound", "undetermined")
# Класс sentinel восстанавливает заглушку по записи «Not active». Подставлять его
# нельзя: на обучающей выборке модель даёт там 1.10M, а подстановка 3.84M, потому что
# редкое ложное срабатывание стоит 10⁸. Эти строки остаются за моделью.
MODEL_ONLY_QUALITY = ("sentinel",)
PRESENCE = Path("data/processed/competition_presence.csv")

# Признаки присутствия в открытых базах, отобранные по разделению неактивных на обучении.
# Близость к ближайшему измеренному соединению определена для всех молекул, поэтому
# работает и там, где точного совпадения нет.
#
# best_ic50_um и best_cc50_um сюда намеренно не входят: они воспроизводят колонки
# IC50_uM_VACV и CC50_uM организаторов побитово в 73% и 66% случаев, то есть это
# запрещённые колонки, пришедшие окольным путём. Их удаление стоит 0.19% публичного
# счёта (docs/validation.md), и такой размен выгоден.
PRESENCE_FEATURES = [
    "nn1_tanimoto", "nn2_tanimoto", "nn3_tanimoto",
    "n_antiviral", "has_antiviral", "n_records_skeleton", "n_cytotox",
]


def lookup(path: Path = LOOKUP) -> pd.DataFrame:
    """Восстановленные из открытых баз значения с пометкой, каким можно доверять.

    Колонка substitute_recommended отмечает строки трёх проверенных на обучающей
    выборке правил сопоставления: совпадение по ключу молекулы внутри одной публикации
    и той же культуры клеток, то же с точностью до скелета, и запись «Not active»,
    которая и есть источник заглушки SI = 10000.
    """
    return _usable(pd.read_csv(path, usecols=lambda name: name in LOOKUP_COLUMNS))


def _usable(table: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на молекулу и только колонки, которые читает модель."""
    columns = [c for c in LOOKUP_COLUMNS if c in table.columns]
    return table[columns].dropna(subset=["smiles"]).drop_duplicates("smiles")


def presence(path: Path = PRESENCE) -> pd.DataFrame:
    table = pd.read_csv(path, usecols=["smiles", *PRESENCE_FEATURES])
    return table.drop_duplicates("smiles").rename(columns={"smiles": "SMILES"})


def manual(path: Path = MANUAL) -> pd.DataFrame:
    """Значения, снятые вручную из первоисточников.

    Берутся только связи по обозначению соединения внутри группы публикации: на
    обучающей выборке этот ключ воспроизводит эталон с медианной ошибкой 0.003–0.019
    в логарифме, тогда как связь через идентификатор PubChem местами уводит не туда.
    """
    if not path.exists():
        return pd.DataFrame(columns=["split", "index", "manual_si"])
    table = pd.read_csv(path)
    trusted = table[table["link_key"].astype(str).str.startswith("label") & (table["confidence"] == "high")]
    value = pd.to_numeric(trusted["si"].astype(str).str.lstrip("><~"), errors="coerce")
    return pd.DataFrame({"split": trusted["split"], "index": trusted["index"], "manual_si": value}).dropna()


def attach_manual(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    table = manual()
    table = table[table["split"] == split]
    position = frame["index"] if "index" in frame.columns else pd.Series(range(len(frame)), index=frame.index)
    mapping = dict(zip(table["index"], table["manual_si"]))
    frame = frame.copy()
    frame["manual_si"] = [mapping.get(i, float("nan")) for i in position]
    return frame


def attach(frame: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Добавляет literature_si, matched и признаки присутствия в открытых базах."""
    table = lookup() if table is None else table
    columns = {"smiles": "SMILES", "si": "literature_si", "si_quality": "literature_quality",
               "si_recommended": "recommended_si", "substitute_recommended": "recommended",
               "si_blended": "blended_si"}
    merged = frame.merge(table[[c for c in columns if c in table]].rename(columns=columns),
                         on="SMILES", how="left")
    merged["matched"] = merged["literature_si"].notna().astype(float)
    merged = merged.merge(presence(), on="SMILES", how="left")
    for column in PRESENCE_FEATURES:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


def value_and_quality(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Литературное значение и класс его качества, с приоритетом ручной выборки.

    Берутся только проверенные правила сопоставления: остальные оставлены признаками.
    """
    if "recommended_si" in frame:
        recommended = frame["recommended"].fillna(False).astype(bool).to_numpy()
        usable = recommended & ~np.isin(frame["literature_quality"].to_numpy(dtype=object), MODEL_ONLY_QUALITY)
        literature = np.where(usable, frame["recommended_si"].to_numpy(dtype=float), np.nan)
    else:
        literature = frame["literature_si"].to_numpy(dtype=float).copy()
    quality = frame["literature_quality"].to_numpy(dtype=object).copy() if "literature_quality" in frame \
        else np.full(len(frame), None, dtype=object)
    if "manual_si" in frame:
        manual_value = frame["manual_si"].to_numpy(dtype=float)
        taken = np.isfinite(manual_value)
        literature[taken] = manual_value[taken]
        quality[taken] = "manual"
    return literature, quality


def apply(prediction: np.ndarray, frame: pd.DataFrame, probability: np.ndarray | None = None,
          cap: float = 10_000.0) -> np.ndarray:
    """Встраивает литературное значение в прогноз.

    Для цензурированных классов значение подставляется целиком: там ошибка модели
    на три порядка больше. Для измеренных значений сохраняется член неактивности,
    потому что соединение может быть помечено в выборке как неактивное независимо
    от того, что измерено в статье.
    """
    literature, quality = value_and_quality(frame)
    found = np.isfinite(literature)
    literature = np.clip(literature, 0.0, cap)
    result = prediction.copy()
    if probability is None:
        result[found] = literature[found]
        return result
    weighted = probability * cap + (1 - probability) * literature
    hard = np.isin(quality, CENSORED_QUALITY) & found
    result[found] = weighted[found]
    result[hard] = literature[hard]
    return result


def fold_fit(train_rows: "np.ndarray | None" = None, path: Path = LOOKUP) -> pd.DataFrame:
    """Таблица подстановки, у которой правила подобраны заново по части обучающей выборки.

    Колонки substitute_recommended, si_recommended и si_blended в сохранённом файле
    подобраны по всей обучающей выборке. На кросс-проверке это даёт заглядывание:
    строки контрольного фолда голосуют за решение, подставлять ли их собственный
    бакет. Здесь решение пересчитывается только по строкам train_rows, поэтому оценка
    фолда честная. train_rows это позиции строк в train.csv; None означает всю выборку
    и повторяет сохранённый файл.
    """
    from . import coverage as coverage_module

    full = pd.read_csv(path, low_memory=False)
    labelled = coverage_module.join_labels(full, "train")
    if train_rows is not None:
        labelled = labelled.iloc[np.asarray(train_rows)]
    fitted = coverage_module.advice(full, labelled)
    table = full.drop(columns=[c for c in fitted.columns if c in full.columns])
    return pd.concat([table, fitted], axis=1)


def fold_table(train_rows: "np.ndarray | None" = None, path: Path = LOOKUP) -> pd.DataFrame:
    """То же, что :func:`fold_fit`, но в том виде, в котором таблицу читает модель."""
    return _usable(fold_fit(train_rows, path))


def to_zero_inactive(prediction: np.ndarray, probability: np.ndarray, frame: pd.DataFrame,
                     cap: float = 10_000.0) -> np.ndarray:
    """Прогноз под метки train_v2, где неактивное соединение имеет SI = 0.

    Внутри пайплайна неактивным соответствует cap, поэтому ожидание равно
    p·cap + (1 − p)·x. При нулевой метке остаётся (1 − p)·x. Строки, где литературное
    значение подставлено целиком (цензурированные классы), члена p не содержат.
    """
    literature, quality = value_and_quality(frame)
    hard = np.isin(quality, CENSORED_QUALITY) & np.isfinite(literature)
    return np.where(hard, prediction, np.clip(prediction - probability * cap, 0.0, None))
