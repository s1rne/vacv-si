"""Интерпретация классификатора неактивности: SHAP, подструктуры, остовы Мурко."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from sklearn.inspection import permutation_importance

from common import frames, probabilities, test_truth
from vacv.features import parse, scaffold
from vacv.literature import PRESENCE_FEATURES
from vacv.pipeline import ContextEncoder, SelectivityModel, structure_matrix

RDLogger.DisableLog("rdApp.*")
MORGAN_BITS = 2048
FIGURES = "docs/figures"


def feature_names() -> list[str]:
    maccs_names = [f"MACCS {i}" for i in range(167)]
    return [f"Morgan {i}" for i in range(MORGAN_BITS)] + maccs_names + ["доля неактивных в группе"] \
        + ["matched", *PRESENCE_FEATURES]


def bit_substructures(frame: pd.DataFrame, bits: list[int]) -> dict[int, str]:
    """SMILES окружения, задающего бит отпечатка Моргана, по первому встреченному примеру."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)
    found: dict[int, str] = {}
    for smiles in frame["SMILES"]:
        mol = parse(smiles)
        if mol is None:
            continue
        info = rdFingerprintGenerator.AdditionalOutput()
        info.AllocateBitInfoMap()
        generator.GetCountFingerprint(mol, additionalOutput=info)
        for bit, environments in info.GetBitInfoMap().items():
            if bit in bits and bit not in found:
                atom, radius = environments[0]
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom) if radius else []
                submol = Chem.PathToSubmol(mol, env) if radius else None
                found[bit] = Chem.MolToSmiles(submol) if submol is not None and submol.GetNumAtoms() \
                    else mol.GetAtomWithIdx(atom).GetSymbol()
        if len(found) == len(bits):
            break
    return found


def main() -> None:
    train, test, y = frames()
    labels = (y == 10_000.0).astype(int)
    model = SelectivityModel(0).fit(train, y)
    names = feature_names()

    matrix = structure_matrix(train)
    features = model._grouped_features(train, matrix)
    explainer = shap.TreeExplainer(model.grouped_)
    values = explainer.shap_values(features)
    if isinstance(values, list):
        values = values[1]
    if values.ndim == 3:
        values = values[:, :, 1]
    strength = np.abs(values).mean(axis=0)
    order = np.argsort(-strength)[:25]

    print("SHAP на модели grouped: 25 признаков с наибольшим средним |вкладом|")
    print(f"{'признак':30} {'|SHAP|':>9} {'знак':>6} {'доля молекул':>13}")
    for position in order:
        present = features[:, position] > (0 if position < MORGAN_BITS + 167 else -np.inf)
        direction = np.sign(np.corrcoef(features[:, position], values[:, position])[0, 1])
        print(f"{names[position]:30} {strength[position]:9.4f} "
              f"{'к неактивным' if direction > 0 else 'к активным':>12} {present.mean():13.1%}")

    chemical = [p for p in order if p < MORGAN_BITS + 167]
    print(f"\nиз них структурных признаков {len(chemical)} из 25; "
          f"суммарный |SHAP| структуры {strength[:MORGAN_BITS + 167].sum():.3f}, "
          f"признака группы {strength[MORGAN_BITS + 167]:.3f}, "
          f"открытых баз {strength[MORGAN_BITS + 168:].sum():.3f}")

    morgan_bits = [int(p) for p in chemical if p < MORGAN_BITS]
    substructures = bit_substructures(train, morgan_bits)
    maccs_smarts = {i: MACCSkeys.smartsPatts.get(i, ("?",))[0] for i in range(167)}
    print("\nрасшифровка структурных признаков:")
    for position in chemical:
        direction = np.sign(np.corrcoef(features[:, position], values[:, position])[0, 1])
        where = "неактивность" if direction > 0 else "активность"
        if position < MORGAN_BITS:
            print(f"  Morgan {position:4d}  {substructures.get(int(position), '?'):28} → {where}")
        else:
            key = int(position) - MORGAN_BITS
            print(f"  MACCS {key:3d}   {maccs_smarts.get(key, '?')[:28]:28} → {where}")

    print("\nперестановочная важность контекстной модели (AUC):")
    encoder = ContextEncoder().fit(train)
    context = encoder.transform(train)
    context_names = [f"strain={v}" for v in encoder.columns_["strain"]] \
        + [f"culture={v}" for v in encoder.columns_["Culture"]] \
        + ["title_len", "title_digits", "title_is_number", "title_has_dash", "title_leading"] \
        + ["matched", *PRESENCE_FEATURES]
    importance = permutation_importance(model.context_, context, labels, n_repeats=10,
                                        random_state=0, scoring="roc_auc")
    for position in np.argsort(-importance.importances_mean)[:12]:
        print(f"  {context_names[position]:36} {importance.importances_mean[position]:+.4f}")

    # Остовы Мурко: что модель считает неактивным и что в литературе оказалось селективным.
    probability, _ = probabilities()
    truth = test_truth()
    scaffolds = pd.Series([scaffold(parse(s)) or "(без цикла)" for s in test["SMILES"]])
    report = pd.DataFrame({
        "scaffold": scaffolds,
        "p": probability,
        "si": truth,
        "matched": test["matched"].to_numpy(),
    })
    grouped = report.groupby("scaffold").agg(строк=("p", "size"), средняя_p=("p", "mean"),
                                             медиана_si=("si", "median"))
    print("\nостовы Мурко, которые модель уверенно относит к неактивным (не меньше 3 молекул):")
    print(grouped[grouped["строк"] >= 3].nlargest(8, "средняя_p").to_string())
    print("\nостовы Мурко самых селективных соединений теста (SI не меньше 100, вне заглушки):")
    selective = report[(report["si"] >= 100) & (report["si"] < 10_000)]
    print(selective.groupby("scaffold").agg(строк=("si", "size"), медиана_si=("si", "median"),
                                            средняя_p=("p", "mean")).nlargest(8, "строк").to_string())

    descriptor_report(train, labels)


def descriptor_report(train: pd.DataFrame, labels: np.ndarray) -> None:
    """Физхимические дескрипторы: чем неактивные отличаются от активных."""
    wanted = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
              "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount"]
    rows = []
    for smiles in train["SMILES"]:
        mol = parse(smiles)
        values = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)
        rows.append({name: values.get(name) for name in wanted})
    table = pd.DataFrame(rows)
    from sklearn.metrics import roc_auc_score
    print("\nдескрипторы: медианы и разделение заглушки (AUC)")
    print(f"{'дескриптор':20} {'неактивные':>11} {'активные':>10} {'AUC':>7}")
    for name in wanted:
        column = table[name].to_numpy(float)
        good = ~np.isnan(column)
        print(f"{name:20} {np.median(column[labels == 1]):11.1f} {np.median(column[labels == 0]):10.1f} "
              f"{roc_auc_score(labels[good], column[good]):7.3f}")


if __name__ == "__main__":
    main()
