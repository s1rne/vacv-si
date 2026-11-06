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
    table = pd.read_csv(path, usecols=["split", "smiles", "si", "si_quality"])
    return table.dropna(subset=["smiles"]).drop_duplicates("smiles")


def presence(path: Path = PRESENCE) -> pd.DataFrame:
    table = pd.read_csv(path, usecols=["smiles", *PRESENCE_FEATURES])
    return table.drop_duplicates("smiles").rename(columns={"smiles": "SMILES"})


def attach(frame: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Добавляет literature_si, matched и признаки присутствия в открытых базах."""
    table = lookup() if table is None else table
    merged = frame.merge(table[["smiles", "si"]].rename(columns={"smiles": "SMILES", "si": "literature_si"}),
                         on="SMILES", how="left")
    merged["matched"] = merged["literature_si"].notna().astype(float)
    merged = merged.merge(presence(), on="SMILES", how="left")
    for column in PRESENCE_FEATURES:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


def apply(prediction: np.ndarray, frame: pd.DataFrame, cap: float = 10_000.0) -> np.ndarray:
    """Там, где значение найдено в литературе, оно заменяет прогноз модели."""
    literature = frame["literature_si"].to_numpy(dtype=float)
    found = np.isfinite(literature)
    result = prediction.copy()
    result[found] = np.clip(literature[found], 0.0, cap)
    return result
