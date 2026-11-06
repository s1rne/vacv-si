"""Химический портрет двух классов: чем неактивные против VACV отличаются от активных.

Нужен для защиты: модель должна опираться на различия, которые химик видит и может
объяснить, а не только на номер публикации. Печатаются три таблицы — физхимия,
остовы Мурко и структурные алерты.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit.Chem import FilterCatalog
from scipy.stats import mannwhitneyu

from experiments.chem_common import DATA, LABELS, SMILES
from vacv.chemistry import PHYSCHEM_NAMES, block_matrix, parse
from vacv.features import scaffold

ALERTS = {"PAINS": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS,
          "Brenk": FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK}


def physchem_table() -> pd.DataFrame:
    matrix = block_matrix(SMILES, "physchem")
    rows = []
    for index, name in enumerate(PHYSCHEM_NAMES):
        column = matrix[:, index]
        active, inactive = column[LABELS == 0], column[LABELS == 1]
        rows.append({
            "признак": name,
            "активные": np.median(active),
            "неактивные": np.median(inactive),
            "p": mannwhitneyu(active, inactive).pvalue,
        })
    return pd.DataFrame(rows).sort_values("p")


def scaffold_table(minimum: int = 5) -> pd.DataFrame:
    keys = [scaffold(parse(text)) or "(ациклический)" for text in SMILES]
    table = pd.DataFrame({"остов": keys, "неактивен": LABELS})
    grouped = table.groupby("остов")["неактивен"].agg(["size", "mean"])
    return grouped[grouped["size"] >= minimum].sort_values("mean", ascending=False)


def alert_table() -> pd.DataFrame:
    rows = {}
    for name, catalog in ALERTS.items():
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(catalog)
        filters = FilterCatalog.FilterCatalog(params)
        hits = np.array([len(filters.GetMatches(parse(text))) for text in SMILES])
        rows[name] = {"активные": (hits[LABELS == 0] > 0).mean(),
                      "неактивные": (hits[LABELS == 1] > 0).mean()}
    return pd.DataFrame(rows).T


def main() -> None:
    y = DATA.y
    print(f"строк {len(y)}, неактивных {LABELS.sum()}, средний SI активных {y[LABELS == 0].mean():.1f}\n")
    print("— физхимия, медианы по классам, критерий Манна—Уитни —")
    print(physchem_table().to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n— остовы Мурко от пяти соединений, доля неактивных —")
    print(scaffold_table().to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n— доля соединений хотя бы с одним алертом —")
    print(alert_table().to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
