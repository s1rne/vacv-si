"""Переносится ли «неактивно против VACV» из внешнего корпуса на заглушку соревнования.

В корпусе признак неактивности задан прямо: `ic50_relation == '>'` означает, что IC50
не определена, известна только нижняя граница. Это в точности смысл заглушки SI = 10000
в файлах соревнования. Вопрос в том, предсказуема ли эта метка по строению молекулы
и предсказуема ли она за пределами той публикации, из которой взята.

Сравниваются четыре сигнала: вероятность с головы неактивности, предсказанная log10 IC50
(непрерывный выход AFT, обученный с учётом цензуры), производный log10 SI с обратным знаком
и простой факт присутствия соединения в открытых базах. Последним текущий пайплайн уже
пользуется, и он служит честным ориентиром для остальных.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from vacv.dataset import load
from vacv.endpoints import load_corpus
from vacv.heads import EndpointHeads, corpus_matrix, inactivity_classifier
from vacv.literature import attach
from vacv.pipeline import INACTIVE
from vacv.splits import fold_indices

from experiments.endpoint_cv import splits


def corpus_auc(corpus: pd.DataFrame) -> dict[str, float]:
    """AUC головы неактивности внутри корпуса при трёх разбиениях."""
    labels = corpus["inactive"].to_numpy(int)
    matrix = corpus_matrix(corpus)
    scores = {}
    for name, folds in splits(corpus).items():
        probability = np.zeros(len(corpus))
        for fold in range(len(folds)):
            train_index, validation_index = fold_indices(folds, fold)
            model = inactivity_classifier().fit(matrix[train_index], labels[train_index])
            probability[validation_index] = model.predict_proba(matrix[validation_index])[:, 1]
        scores[name] = roc_auc_score(labels, probability)
    return scores


def main() -> None:
    corpus = load_corpus()
    print("AUC головы неактивности внутри корпуса (метка: IC50 со знаком «>»)")
    for name, value in corpus_auc(corpus).items():
        print(f"  {name:12} {value:.3f}")

    data = load()
    y = data.y
    labels = (y == INACTIVE).astype(int)
    frame = attach(data.train)
    heads = EndpointHeads().fit(corpus)
    predicted = heads.predict(data.train)

    scores = {
        "p_inactive с головы": predicted["p_inactive"].to_numpy(float),
        "предсказанная log10 IC50": predicted["log_ic50"].to_numpy(float),
        "минус производный log10 SI": -predicted["log_si"].to_numpy(float),
        "найдено в открытых базах": -frame["matched"].to_numpy(float),
    }
    unseen = frame["matched"].to_numpy(float) == 0
    print(f"\nперенос на выборку соревнования: заглушек {labels.sum()} из {len(labels)}, "
          f"нет в открытых базах {int(unseen.sum())}")
    print(f"{'сигнал':28} {'AUC все':>8} {'AUC вне баз':>12}")
    for name, score in scores.items():
        outside = roc_auc_score(labels[unseen], score[unseen]) if labels[unseen].sum() else float("nan")
        print(f"{name:28} {roc_auc_score(labels, score):8.3f} {outside:12.3f}")

    active = labels == 0
    log_y = np.log10(np.clip(y[active], 1e-3, None))
    log_si = predicted.loc[active, "log_si"].to_numpy(float)
    found = frame["matched"].to_numpy(float)[active] == 1
    print("\nпроизводный log10 SI на активных: поиск против предсказания")
    print(f"{'подвыборка':28} {'n':>5} {'Pearson':>8} {'RMSE':>7}")
    for name, mask in (("есть в открытых базах", found), ("нет в открытых базах", ~found)):
        design = np.column_stack([np.ones(mask.sum()), log_si[mask]])
        fitted = design @ np.linalg.lstsq(design, log_y[mask], rcond=None)[0]
        print(f"{name:28} {mask.sum():5d} {np.corrcoef(log_si[mask], log_y[mask])[0, 1]:8.3f} "
              f"{np.sqrt(np.mean((fitted - log_y[mask]) ** 2)):7.3f}")


if __name__ == "__main__":
    main()
