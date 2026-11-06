"""Главная таблица: химия отдельно, метаданные отдельно, открытые данные отдельно, вместе.

Три источника сигнала разделены намеренно.

- `chemistry` — только строение молекулы из SMILES, никаких внешних колонок;
- `metadata` — штамм, культура, обозначение соединения и доля неактивных в группе
  strain+Culture, то есть косвенная ссылка на публикацию-источник;
- `opendata` — присутствие молекулы в ChEMBL и PubChem и близость к измеренным
  соединениям; это внешние данные, частично структурные по природе.

Ансамбль собирается взвешенным средним с весами, обратными OOF-MSE (docs/research.md,
раздел 2.6: ridge-стекинг при таком N переобучается). Веса для каждого фолда считаются
по остальным фолдам, чтобы не настраивать их на тот же контроль.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.chem_common import DATA, FOLDS, LABELS, Y, classifier_oof, matrix
from vacv.chemistry import RECOMMENDED
from vacv.evaluate import competition_mse
from vacv.literature import PRESENCE_FEATURES, attach
from vacv.pipeline import ContextEncoder, group_rate
from vacv.splits import fold_indices

SPLITS = ("random", "scaffold")

# Разнообразие берётся по двум осям: разные блоки признаков с одной моделью и разные
# модели на отобранной конкатенации. По docs/research.md, раздел 2.6, выигрыш даёт
# прежде всего набор блоков, поэтому блоков в составе больше, чем моделей.
CHEMISTRY_PARTS = (
    ("extra / rdkit2d", "extra", ("rdkit2d",)),
    ("extra / erg", "extra", ("erg",)),
    ("extra / avalon", "extra", ("avalon",)),
    ("extra / maccs", "extra", ("maccs",)),
    ("extra / ecfp2", "extra", ("ecfp2",)),
    ("extra / physchem", "extra", ("physchem",)),
    ("extra / отобранные", "extra", RECOMMENDED),
    ("lgbm / отобранные", "lgbm", RECOMMENDED),
    ("svc_rbf / отобранные", "svc_rbf", RECOMMENDED),
    ("svc_tanimoto / ecfp2+avalon", "svc_tanimoto", ("ecfp2", "avalon")),
)


def context_matrix(frame: pd.DataFrame, folds: list[np.ndarray]) -> np.ndarray:
    """Дамми штамма и культуры, признаки обозначения, доля неактивных в группе.

    Кодировка группы считается по обучающей части каждого фолда: доля неактивных —
    это целевое кодирование, и по всей выборке оно протекает.
    """
    encoder = ContextEncoder().fit(frame)
    base = encoder.transform(frame)
    rate = np.zeros(len(frame))
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        rate[validation_index] = group_rate(frame.iloc[train_index], LABELS[train_index],
                                            frame.iloc[validation_index])
    return np.hstack([base, rate[:, None]])


def opendata_matrix(frame: pd.DataFrame) -> np.ndarray:
    merged = attach(frame)
    columns = ["matched", *PRESENCE_FEATURES]
    return merged[columns].to_numpy(dtype=float, na_value=-1.0)


def nested_weights(probability: np.ndarray, train_index: np.ndarray) -> np.ndarray:
    """Веса, обратные MSE условного среднего на обучающей части фолда."""
    level = Y[train_index][LABELS[train_index] == 0].mean()
    errors = np.array([np.mean((p[train_index] * 10_000 + (1 - p[train_index]) * level - Y[train_index]) ** 2)
                       for p in probability])
    weights = 1.0 / np.maximum(errors, 1.0)
    return weights / weights.sum()


def ensemble(probability: list[np.ndarray], folds: list[np.ndarray]) -> np.ndarray:
    blended = np.zeros(len(Y))
    stacked = np.vstack(probability)
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        weights = nested_weights(stacked, train_index)
        blended[validation_index] = weights @ stacked[:, validation_index]
    return blended


def line(label: str, probability: dict[str, np.ndarray]) -> None:
    scores = []
    for split in SPLITS:
        scores += [roc_auc_score(LABELS, probability[split]),
                   competition_mse(probability[split], Y, FOLDS[split])]
    print(f"{label:38} {scores[0]:7.3f} {scores[1]:12,.0f} {scores[2]:7.3f} {scores[3]:12,.0f}")


def main() -> None:
    frame = DATA.train
    print(f"{'источник / модель':38} {'AUC rn':>7} {'MSE rn':>12} {'AUC sc':>7} {'MSE sc':>12}")

    chemistry: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    for label, model, blocks in CHEMISTRY_PARTS:
        probability = {split: classifier_oof(model, matrix(blocks), FOLDS[split]) for split in SPLITS}
        line(label, probability)
        for split in SPLITS:
            chemistry[split].append(probability[split])

    single = CHEMISTRY_PARTS.index(next(part for part in CHEMISTRY_PARTS if part[2] == RECOMMENDED))
    aggregated = {
        "chemistry": {split: ensemble(chemistry[split], FOLDS[split]) for split in SPLITS},
        "single": {split: chemistry[split][single] for split in SPLITS},
    }
    line("химия / ансамбль 1/MSE", aggregated["chemistry"])

    sources = {"metadata": {split: context_matrix(frame, FOLDS[split]) for split in SPLITS},
               "opendata": {split: opendata_matrix(frame) for split in SPLITS}}
    for source in ("metadata", "opendata"):
        aggregated[source] = {split: classifier_oof("extra", sources[source][split], FOLDS[split])
                              for split in SPLITS}
        line(f"{source} / extra", aggregated[source])

    best = {}
    for source, label in (("single", "отобранная химия"), ("chemistry", "ансамбль химии")):
        print()
        for weight in np.linspace(0.0, 1.0, 11):
            probability = {split: weight * aggregated[source][split]
                           + (1 - weight) * aggregated["metadata"][split] for split in SPLITS}
            line(f"{label} {weight:.1f} + metadata {1 - weight:.1f}", probability)
            mse = competition_mse(probability["random"], Y, FOLDS["random"])
            if source not in best or mse < best[source][0]:
                best[source] = (mse, weight)

    print()
    weight = best["single"][1]
    for share in (0.0, 0.1, 0.2, 0.3):
        probability = {}
        for split in SPLITS:
            pair = weight * aggregated["single"][split] + (1 - weight) * aggregated["metadata"][split]
            probability[split] = (1 - share) * pair + share * aggregated["opendata"][split]
        line(f"лучшая пара + opendata {share:.1f}", probability)


if __name__ == "__main__":
    main()
