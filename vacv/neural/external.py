"""Внешний набор по VACV как задача предобучения.

`data/processed/vacv_si.csv` — 1308 соединений, собранных из ChEMBL и PubChem, с IC50, CC50
и DOI. Это легальные внешние данные: правила запрещают только колонки файлов соревнования.

Три цели предобучения:

- `log_ic50`, `log_cc50` — в логарифме, только по незацензурированным значениям;
- `inactive` — отношение «IC50 > граница» означает, что активность не достигнута на
  максимальной протестированной концентрации. Это ровно та метка, которую предсказывает
  классификатор соревнования, только на другой выборке. 754 из 1308 записей цензурированы
  сверху, так что метка не вырожденная.

Валидационные соединения выкидываются из предобучения по скелетному InChIKey: иначе сеть
увидит IC50 той молекулы, SI которой её же потом просят предсказать, и оценка станет
нечестной. Совпадений с обучающей выборкой соревнования 58 по SMILES.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

TABLE = Path("data/processed/vacv_si.csv")
TARGETS = ("log_ic50", "log_cc50", "inactive")
UNCENSORED = ("=",)
CENSORED_ABOVE = (">", ">=")


def skeleton(smiles: str) -> str:
    """Первые 14 символов InChIKey: соль, изотоп и стереохимия игнорируются."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol)[:14]
    except Exception:
        return ""


def keys_of(frame: pd.DataFrame) -> list[str]:
    return [skeleton(value) for value in frame["SMILES"]]


def table(path: Path = TABLE) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["smiles", "ic50", "cc50", "inchikey14",
                                       "ic50_relation", "cc50_relation"])
    frame = frame.dropna(subset=["smiles"]).drop_duplicates("smiles").reset_index(drop=True)
    frame["log_ic50"] = np.where(frame["ic50_relation"].isin(UNCENSORED),
                                 np.log10(frame["ic50"].clip(lower=1e-4)), np.nan)
    frame["log_cc50"] = np.where(frame["cc50_relation"].isin(UNCENSORED),
                                 np.log10(frame["cc50"].clip(lower=1e-4)), np.nan)
    frame["inactive"] = frame["ic50_relation"].isin(CENSORED_ABOVE).astype(float)
    return frame


def pretraining(exclude: set[str], frame: pd.DataFrame | None = None,
                standardize: bool = True) -> tuple[list[str], np.ndarray]:
    """SMILES и матрица целей внешнего набора без соединений из `exclude`."""
    frame = table() if frame is None else frame
    keep = ~frame["inchikey14"].fillna("").str[:14].isin(exclude)
    kept = frame[keep].reset_index(drop=True)
    targets = kept[list(TARGETS)].to_numpy(dtype=np.float64)
    if standardize:
        for column in (0, 1):
            values = targets[:, column]
            known = np.isfinite(values)
            if known.sum() > 1:
                targets[known, column] = (values[known] - values[known].mean()) / values[known].std()
    return list(kept["smiles"]), targets
