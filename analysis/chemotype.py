"""Проверка химического вывода: что отличает активные серии от неактивных.

Утверждение к защите — активность связана с арилкарбоксамидом или ацилгидразидом
на насыщенном каркасе, а те же каркасы в виде простого сложного эфира неактивны.
Здесь оно проверяется счётом по SMARTS на обучении и на восстановленных метках теста.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from common import frames, oracle

RDLogger.DisableLog("rdApp.*")

CAGE = Chem.MolFromSmarts("[$([CX4]1[CX4][CX4][CX4]([CX4]1)),$(C12CC3CC(C1)CC(C2)C3),$(C1CC2CCC1C2)]")
PATTERNS = {
    "ацилгидразид / гидразон C(=O)N-N": "[CX3](=O)[NX3][NX3,NX2]",
    "арилкарбоксамид c-C(=O)N": "c[CX3](=O)[NX3]",
    "сложный эфир C(=O)O-C": "[CX3](=O)[OX2][CX4]",
    "мочевина N-C(=O)-N": "[NX3][CX3](=O)[NX3]",
    "насыщенный каркас (норборнан/адамантан)": "[$(C12CC3CC(C1)CC(C2)C3),$(C1CC2CCC1C2)]",
    "незамещённый бифенил/дифенилметан": "c1ccccc1[CX4H2]c1ccccc1",
}


def table(frame: pd.DataFrame, si: np.ndarray, title: str) -> None:
    molecules = [Chem.MolFromSmiles(s) for s in frame["SMILES"]]
    sentinel = si == 10_000.0
    active = ~sentinel & (si > 0)
    print(f"\n{title} ({len(si)} строк, заглушка {sentinel.mean():.1%})")
    print(f"{'подструктура':44} {'молекул':>8} {'доля заглушки':>14} {'медиана SI':>12}")
    for name, smarts in PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts)
        hit = np.array([m is not None and m.HasSubstructMatch(pattern) for m in molecules])
        if not hit.any():
            continue
        median = np.median(si[hit & active]) if (hit & active).any() else np.nan
        print(f"{name:44} {hit.sum():8d} {sentinel[hit].mean():14.1%} {median:12.1f}")


def main() -> None:
    train, test, y = frames()
    table(train, y, "обучение")
    table(test, oracle(pd.read_csv("data/raw/kaggle/test.csv")), "тест, метки восстановлены")


if __name__ == "__main__":
    main()
