"""Домен применимости: три определения и их связь с ошибкой.

Структурный домен (близость к обучающим молекулам) здесь не работает так, как ожидает
OECD-рамка, и это важный честный результат. Решает не новизна скелета, а две другие
вещи: известна ли группа strain+Culture и есть ли о соединении запись в открытых базах.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import frames, oracle, probabilities


def main() -> None:
    train, test, y = frames()
    raw_train = pd.read_csv("data/raw/kaggle/train.csv")
    raw_test = pd.read_csv("data/raw/kaggle/test.csv")
    truth = oracle(raw_test)
    submission = pd.read_csv("outputs/submission.csv").set_index("index").loc[raw_test["index"]]["SI VACV"].to_numpy()
    squared = (submission - truth) ** 2
    nearest = np.load("analysis/cache/test_nearest_train.npy")
    probability, _ = probabilities()

    group = lambda f: f[["strain", "Culture"]].astype(str).agg(" | ".join, axis=1)
    known_group = group(raw_test).isin(set(group(raw_train))).to_numpy()
    evidence = (test["n_records_skeleton"].fillna(0).to_numpy() > 0)
    matched = test["matched"].to_numpy() == 1
    sentinel = truth == 10_000.0

    print("определение 1: структурная близость (Tanimoto к ближайшему обучающему)")
    print(f"  порог берётся как 5-й процентиль таких же расстояний внутри обучения")
    # Порог по обучению: для каждой обучающей молекулы — её ближайший сосед в обучении.
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in raw_train["SMILES"]]
    within = []
    for position, fingerprint in enumerate(train_fingerprints):
        similarity = np.array(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fingerprints))
        similarity[position] = -1
        within.append(similarity.max())
    threshold = float(np.percentile(within, 5))
    inside = nearest >= threshold
    print(f"  порог {threshold:.3f}; внутри домена {inside.sum()} из {len(inside)} ({inside.mean():.1%})")
    print(f"  MSE внутри {squared[inside].mean():,.0f}, снаружи {squared[~inside].mean():,.0f}")
    print(f"  доля заглушки внутри {sentinel[inside].mean():.1%}, снаружи {sentinel[~inside].mean():.1%}")
    print(f"  медианная |ошибка| log10 на активных: внутри "
          f"{np.median(np.abs(np.log10(np.clip(submission[inside & ~sentinel], 1e-6, None)) - np.log10(np.clip(truth[inside & ~sentinel], 1e-6, None)))):.2f}, "
          f"снаружи {np.median(np.abs(np.log10(np.clip(submission[~inside & ~sentinel], 1e-6, None)) - np.log10(np.clip(truth[~inside & ~sentinel], 1e-6, None)))):.2f}")

    print("\nопределение 2: известна ли группа strain+Culture")
    print(f"  внутри {known_group.sum()} из {len(known_group)} ({known_group.mean():.1%})")
    print(f"  MSE внутри {squared[known_group].mean():,.0f}, снаружи {squared[~known_group].mean():,.0f}")

    print("\nопределение 3: есть ли запись об этом соединении в открытых базах")
    for name, mask in (("точное значение SI восстановлено", matched),
                       ("есть хоть одна запись", evidence & ~matched),
                       ("записей нет", ~evidence)):
        print(f"  {name:34} {mask.sum():4d} строк  MSE {squared[mask].mean():12,.0f}  "
              f"доля заглушки {sentinel[mask].mean():6.1%}")

    print("\nсовместный домен (группа известна и значение восстановлено):")
    combined = known_group & matched
    print(f"  внутри {combined.sum()} ({combined.mean():.1%}), MSE {squared[combined].mean():,.0f}")
    print(f"  снаружи {int((~combined).sum())}, MSE {squared[~combined].mean():,.0f}")

    print("\nчем объясняется, что структурный домен не работает:")
    frame = pd.DataFrame({"nearest": nearest, "matched": matched, "sentinel": sentinel})
    bins = pd.cut(frame["nearest"], [0, 0.3, 0.5, 0.7, 0.9, 1.01])
    print(frame.groupby(bins, observed=True).agg(
        строк=("matched", "size"), доля_с_литературой=("matched", "mean"),
        доля_заглушки=("sentinel", "mean")).to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nнадёжность как функция уверенности классификатора:")
    confidence = np.abs(probability - 0.5)
    frame = pd.DataFrame({"уверенность": confidence, "квадрат": squared})
    bins = pd.cut(frame["уверенность"], [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5])
    print(frame.groupby(bins, observed=True).agg(
        строк=("квадрат", "size"), MSE=("квадрат", "mean")).to_string(float_format=lambda v: f"{v:,.0f}"))

    np.save("analysis/cache/test_inside_structural.npy", inside)
    np.save("analysis/cache/test_known_group.npy", known_group)
    pd.Series({"structural_threshold": threshold}).to_csv("analysis/domain_threshold.csv")


if __name__ == "__main__":
    main()
