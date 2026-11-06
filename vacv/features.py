"""Признаки молекулы из SMILES.

Набор намеренно смешанный: дескрипторы RDKit описывают физхимию (растворимость,
липофильность, размер), счётные отпечатки Моргана описывают структурные фрагменты.
Градиентный бустинг на таких данных обычно сильнее одного типа признаков.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

DESCRIPTORS = [name for name, _ in Descriptors.descList]


def parse(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None


def descriptors(mol: Chem.Mol) -> np.ndarray:
    values = np.fromiter(Descriptors.CalcMolDescriptors(mol, missingVal=np.nan).values(), dtype=np.float64)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def morgan_counts(mol: Chem.Mol, radius: int = 2, size: int = 2048) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
    return np.asarray(generator.GetCountFingerprintAsNumPy(mol), dtype=np.float32)


def maccs(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)


def scaffold(mol: Chem.Mol) -> str:
    """Остов Мурко: по нему делятся выборки, чтобы не мерить запоминание серий аналогов."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return ""
