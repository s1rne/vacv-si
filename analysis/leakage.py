"""Аудит утечек: откуда берётся сигнал и что из него не обобщается."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from common import frames, oracle

RDLogger.DisableLog("rdApp.*")
LOOKUP = "data/processed/competition_open_data_lookup.csv"
PRESENCE = "data/processed/competition_presence.csv"


def fingerprints(smiles: pd.Series):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        out.append(generator.GetFingerprint(mol) if mol is not None else None)
    return out


def inchikeys(smiles: pd.Series) -> list[str | None]:
    out = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        out.append(Chem.MolToInchiKey(mol) if mol is not None else None)
    return out


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    train, test, y = frames()
    raw_train = pd.read_csv("data/raw/kaggle/train.csv")
    raw_test = pd.read_csv("data/raw/kaggle/test.csv")
    truth = oracle(raw_test)
    lookup = pd.read_csv(LOOKUP)

    section("1. Происхождение литературных значений")
    document = lookup["document_id"].astype(str)
    print(f"строк с восстановленным SI: {lookup['si'].notna().sum()} из {len(lookup)}")
    print(f"из них document_id начинается с KAGGLE: {int(document.str.startswith('KAGGLE').sum())}")
    print(f"уникальных публикаций-источников: {document[lookup['si'].notna()].nunique()}")
    print(f"строк с DOI: {int(lookup.loc[lookup['si'].notna(), 'doi'].notna().sum())}")
    print(f"правила отбора: {lookup['rule'].value_counts().to_dict()}")

    for name, split, reference in (("train", "train", raw_train["SI VACV"].to_numpy(float)),
                                   ("test", "test", truth)):
        side = lookup[lookup["split"].eq(split)].reset_index(drop=True)
        found = side["si"].notna().to_numpy()
        target = reference[found]
        value = side.loc[found, "si"].to_numpy(float)
        usable = (target > 0) & (target != 10_000.0)
        identical = np.isclose(value[usable], target[usable], rtol=1e-6)
        error = np.log10(value[usable]) - np.log10(target[usable])
        print(f"\n{name}: {found.sum()} совпадений, из них сравнимых {usable.sum()}")
        print(f"  побитово совпадает с меткой организаторов: {identical.mean():.1%}")
        print(f"  медианная |ошибка| log10 {np.median(np.abs(error)):.3f}, в пределах 2x {np.mean(np.abs(error) <= np.log10(2)):.1%}")
        print(f"  найдено, но метка 10000: {int((reference[found] == 10_000.0).sum())} строк")
        print(f"  не найдено, метка 10000: {int((reference[~found] == 10_000.0).sum())} из {int((~found).sum())}")

    section("2. Признаки присутствия против запрещённых колонок")
    presence = pd.read_csv(PRESENCE)
    side = presence[presence["split"].eq("train")].reset_index(drop=True)
    for column, forbidden in (("best_ic50_um", "IC50_uM_VACV"), ("best_cc50_um", "CC50_uM")):
        pair = pd.DataFrame({"open": pd.to_numeric(side[column], errors="coerce"),
                             "comp": pd.to_numeric(raw_train[forbidden], errors="coerce")}).dropna()
        error = np.log10(pair["open"]) - np.log10(pair["comp"])
        print(f"{column} против {forbidden}: {len(pair)} пар, "
              f"побитово {np.isclose(pair['open'], pair['comp'], rtol=1e-6).mean():.1%}, "
              f"медианная |ошибка| log10 {error.abs().median():.3f}, "
              f"Spearman {pair['open'].corr(pair['comp'], method='spearman'):+.2f}")
    print(f"\nиспользуемые колонки nn*_si (SI соседа) в модели: "
          f"{'нет' if 'nn1_si' not in train.columns else 'ДА — это утечка'}")

    section("3. Кодирование группы strain+Culture")
    group = raw_train[["strain", "Culture"]].astype(str).agg(" | ".join, axis=1)
    label = (y == 10_000.0).astype(int)
    stats = pd.DataFrame({"group": group, "label": label}).groupby("group")["label"].agg(["size", "mean"])
    print(f"групп на обучении {len(stats)}, медианный размер {stats['size'].median():.0f}")
    print(f"групп, где метка одинакова у всех строк: {int(stats['mean'].isin([0.0, 1.0]).sum())} из {len(stats)}")
    print(f"доля обучающих строк в таких группах: {stats.loc[stats['mean'].isin([0.0, 1.0]), 'size'].sum() / len(raw_train):.1%}")
    print("\nвклад строки в собственное кодирование (сглаживание 5):")
    print(f"  медианный вес собственной метки 1/(n+5) = {1 / (stats['size'].median() + 5):.3f}")
    print(f"  в самой маленькой группе (n={stats['size'].min():.0f}) = {1 / (stats['size'].min() + 5):.3f}")

    section("4. Дубликаты и почти-дубликаты между обучением и тестом")
    train_keys = inchikeys(raw_train["SMILES"])
    test_keys = inchikeys(raw_test["SMILES"])
    exact = pd.Series(test_keys).isin({k for k in train_keys if k}).to_numpy()
    skeleton = pd.Series([k[:14] if k else None for k in test_keys]).isin(
        {k[:14] for k in train_keys if k}).to_numpy()
    print(f"тестовых строк с точно такой же молекулой в обучении: {int(exact.sum())}")
    print(f"тестовых строк с тем же остовом InChIKey (первые 14): {int(skeleton.sum())}")

    train_fingerprints = [f for f in fingerprints(raw_train["SMILES"]) if f is not None]
    nearest = []
    for fingerprint in fingerprints(raw_test["SMILES"]):
        if fingerprint is None:
            nearest.append(np.nan)
            continue
        nearest.append(max(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fingerprints)))
    nearest = np.array(nearest)
    np.save("analysis/cache/test_nearest_train.npy", nearest)
    for threshold in (0.99, 0.95, 0.9, 0.8, 0.7, 0.5):
        mask = nearest >= threshold
        print(f"  Tanimoto к ближайшему обучающему >= {threshold}: {int(mask.sum()):3d} строк "
              f"({mask.mean():5.1%}), доля заглушки {(truth[mask] == 10_000.0).mean():5.1%}")
    print(f"\nмедианный Tanimoto к ближайшему обучающему: {np.nanmedian(nearest):.3f}")

    submission = pd.read_csv("outputs/submission.csv").set_index("index").loc[raw_test["index"]]["SI VACV"].to_numpy()
    squared = (submission - truth) ** 2
    print("\nошибка в зависимости от близости к обучению:")
    print(f"{'интервал Tanimoto':>20} {'строк':>6} {'MSE':>14} {'доля SSE':>9}")
    bins = [0, 0.3, 0.4, 0.5, 0.7, 0.9, 1.01]
    for low, high in zip(bins, bins[1:]):
        mask = (nearest >= low) & (nearest < high)
        if mask.any():
            print(f"{low:8.2f}–{high:<10.2f} {int(mask.sum()):6d} {squared[mask].mean():14,.0f} "
                  f"{squared[mask].sum() / squared.sum():9.1%}")


if __name__ == "__main__":
    main()
