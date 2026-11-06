"""Случайные SMILES одной и той же молекулы.

Одна молекула записывается множеством корректных строк. Для SMILES-трансформера это
единственный дешёвый способ размножить 320 примеров: обучающая строка каждую эпоху новая,
а на предсказании несколько строк усредняются (test-time augmentation).
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem


def randomized(smiles: str, count: int, seed: int = 0) -> list[str]:
    """`count` различных записей молекулы. Канонический вариант всегда первый."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * count
    out = [Chem.MolToSmiles(mol)]
    rng = np.random.default_rng(seed)
    order = np.arange(mol.GetNumAtoms())
    for _ in range(count * 4):
        if len(out) >= count:
            break
        rng.shuffle(order)
        variant = Chem.MolToSmiles(Chem.RenumberAtoms(mol, [int(i) for i in order]), canonical=False)
        if variant not in out:
            out.append(variant)
    while len(out) < count:
        out.append(out[len(out) % max(1, len(out))])
    return out[:count]


def expand(smiles: list[str], labels: np.ndarray, count: int, seed: int = 0) -> tuple[list[str], np.ndarray]:
    """Обучающая выборка, размноженная в `count` раз, с теми же метками."""
    texts, targets = [], []
    for index, (text, label) in enumerate(zip(smiles, labels)):
        for variant in randomized(text, count, seed=seed + index):
            texts.append(variant)
            targets.append(label)
    return texts, np.asarray(targets)
