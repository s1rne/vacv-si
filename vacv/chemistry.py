"""Химические признаки молекулы и структурная модель неактивности.

Набор блоков повторяет рекомендацию из docs/research.md, раздел 2.6: жадное наращивание
у Kamuntavičius 2025 даёт rdkit_desc → +erg → +ecfp4 → +avalon, и выигрыш даёт именно
набор блоков, а не выбор модели. Блоки считаются по отдельности и склеиваются по запросу,
чтобы их вклад можно было измерить независимо (experiments/chemistry.py).

Два дескриптора RDKit исключены как численно неустойчивые: `Ipc` даёт значения до 1e+162
на длинных цепях (rdkit#1527), `Kappa3` возвращает больше 9000 на очень маленьких
молекулах. Остальные пропуски и бесконечности заменяются нулём.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Crippen, Descriptors, MACCSkeys, QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem import rdReducedGraphs

RDLogger.DisableLog("rdApp.*")

UNSTABLE = ("Ipc", "Kappa3")
DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList if name not in UNSTABLE]
_DESCRIPTOR_FUNCS = [function for name, function in Descriptors.descList if name not in UNSTABLE]

PHYSCHEM_NAMES = [
    "mw", "logp", "tpsa", "hbd", "hba", "rotatable", "rings", "aromatic_rings",
    "heavy_atoms", "heteroatoms", "fraction_csp3", "qed", "molar_refractivity",
    "stereocentres", "formal_charge", "halogens", "spiro", "bridgehead",
    "lipinski_violations", "veber_pass", "ligand_efficiency_proxy",
]


def _morgan(radius: int, size: int, features: bool) -> Callable[[Chem.Mol], np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=size, atomInvariantsGenerator=(
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if features else None))
    return lambda mol: np.asarray(generator.GetCountFingerprintAsNumPy(mol), dtype=np.float32)


def rdkit2d(mol: Chem.Mol) -> np.ndarray:
    return np.fromiter((function(mol) for function in _DESCRIPTOR_FUNCS),
                       dtype=np.float64, count=len(_DESCRIPTOR_FUNCS))


def erg(mol: Chem.Mol) -> np.ndarray:
    """Сокращённый граф Stiefl 2006: 315 чисел, 2D-фармакофор вместо точных фрагментов."""
    return np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32)


def avalon_counts(mol: Chem.Mol, size: int = 1024) -> np.ndarray:
    fingerprint = pyAvalonTools.GetAvalonCountFP(mol, nBits=size)
    vector = np.zeros(size, dtype=np.float32)
    for index, value in fingerprint.GetNonzeroElements().items():
        vector[index] = value
    return vector


def maccs(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)


def physchem(mol: Chem.Mol) -> np.ndarray:
    """Физхимия и правила лекарственного подобия — тот блок, который читает химик."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return np.array([
        mw, logp, tpsa, hbd, hba, rotatable,
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        heavy,
        sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in ("C", "H")),
        rdMolDescriptors.CalcFractionCSP3(mol),
        QED.qed(mol),
        Crippen.MolMR(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)),
        Chem.GetFormalCharge(mol),
        sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ("F", "Cl", "Br", "I")),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        violations,
        float(rotatable <= 10 and tpsa <= 140),
        logp / max(heavy, 1),
    ], dtype=np.float64)


BLOCKS: dict[str, Callable[[Chem.Mol], np.ndarray]] = {
    "rdkit2d": rdkit2d,
    "erg": erg,
    "avalon": avalon_counts,
    "ecfp2": _morgan(2, 1024, features=False),
    "ecfp3": _morgan(3, 1024, features=False),
    "fcfp2": _morgan(2, 1024, features=True),
    "fcfp3": _morgan(3, 1024, features=True),
    "maccs": maccs,
    "physchem": physchem,
}

# Отобрано жадным наращиванием в experiments/chem_blocks.py. Случайное разбиение
# выбирает maccs+rdkit2d+physchem, ErG добавлен за остовное разбиение: в одиночку он даёт
# там лучший AUC 0.880, потому что сокращённый граф описывает фармакофор, а не конкретные
# фрагменты, и переносится на незнакомые серии.
RECOMMENDED = ("rdkit2d", "maccs", "physchem", "erg")


def parse(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"не разбирается SMILES: {smiles}")
    return mol


def block_matrix(smiles: Iterable[str], name: str) -> np.ndarray:
    """Один блок признаков. Пропуски и бесконечности зануляются сразу на выходе."""
    function = BLOCKS[name]
    matrix = np.vstack([np.asarray(function(parse(text)), dtype=np.float64) for text in smiles])
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def features(smiles: Sequence[str], blocks: Sequence[str] = RECOMMENDED) -> np.ndarray:
    return np.hstack([block_matrix(smiles, name) for name in blocks])
