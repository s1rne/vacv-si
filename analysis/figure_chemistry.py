"""Рисунок с химической интерпретацией: вклады подструктур и остовы Мурко."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import frames, probabilities, test_truth
from interpret import MORGAN_BITS, bit_substructures, feature_names
from vacv.features import parse, scaffold
from vacv.pipeline import SelectivityModel, structure_matrix

FIGURES = Path("docs/figures")
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
INK, ACCENT = "#1f3b5c", "#c2432f"


def main() -> None:
    train, test, y = frames()
    model = SelectivityModel(0).fit(train, y)
    features = model._grouped_features(train, structure_matrix(train))
    values = shap.TreeExplainer(model.grouped_).shap_values(features)
    if isinstance(values, list):
        values = values[1]
    if values.ndim == 3:
        values = values[:, :, 1]
    names = feature_names()

    chemical = [p for p in np.argsort(-np.abs(values).mean(axis=0)) if p < MORGAN_BITS + 167][:12]
    substructures = bit_substructures(train, [int(p) for p in chemical if p < MORGAN_BITS])
    from rdkit.Chem import MACCSkeys
    labels, contributions = [], []
    for position in chemical:
        signed = np.sign(np.corrcoef(features[:, position], values[:, position])[0, 1])
        text = substructures.get(int(position), "?") if position < MORGAN_BITS \
            else MACCSkeys.smartsPatts.get(int(position) - MORGAN_BITS, ("?",))[0][:24]
        labels.append(f"{names[position]}  {text}")
        contributions.append(signed * np.abs(values[:, position]).mean())

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                         gridspec_kw={"width_ratios": [1.15, 1]})
    order = np.argsort(contributions)
    colours = [ACCENT if contributions[i] > 0 else "#2e7d5b" for i in order]
    left.barh(range(len(order)), [contributions[i] for i in order], color=colours)
    left.set_yticks(range(len(order)), [labels[i] for i in order], fontsize=7)
    left.axvline(0, color="#555", lw=0.8)
    left.set_xlabel("средний |SHAP|, знак — в сторону неактивности (красный)")
    left.set_title("Структурные признаки классификатора неактивности")

    probability, _ = probabilities()
    truth = test_truth()
    report = pd.DataFrame({
        "scaffold": [scaffold(parse(s)) or "(без цикла)" for s in test["SMILES"]],
        "p": probability, "si": truth,
    })
    selective = report[(report["si"] >= 100) & (report["si"] < 10_000)]
    top = selective.groupby("scaffold").agg(n=("si", "size"), si=("si", "median")).nlargest(6, "n")
    inactive = report[report["si"] == 10_000.0].groupby("scaffold").size().nlargest(6)
    right.barh(range(len(top)), np.log10(top["si"]), color="#2e7d5b")
    right.set_yticks(range(len(top)), [f"{s[:42]}… (n={n})" if len(s) > 42 else f"{s} (n={n})"
                                       for s, n in zip(top.index, top["n"])], fontsize=6.5)
    right.set_xlabel("медианный log10 SI")
    right.set_title("Остовы Мурко самых селективных соединений теста")

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / "chemistry.png", bbox_inches="tight")
    print("docs/figures/chemistry.png")
    print("\nнаиболее частые остовы среди соединений с меткой 10000:")
    print(inactive.to_string())


if __name__ == "__main__":
    main()
