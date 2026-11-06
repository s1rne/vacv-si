"""Вклад каждого блока химических признаков: по одному и жадным наращиванием.

Вопрос: насколько сильна структура сама по себе, без штамма, культуры и обозначения
соединения. В текущем пайплайне структурный блок имеет вес 0, и это главное слабое
место решения перед химической экспертизой.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.chem_common import FOLDS, LABELS, Y, classifier_oof, matrix
from vacv.chemistry import BLOCKS
from vacv.evaluate import competition_mse

MODELS = ("extra", "cat")
HEADER = f"{'блок':34} {'модель':8} {'AUC rn':>7} {'MSE rn':>12} {'AUC sc':>7} {'MSE sc':>12}"


def row(label: str, names, model: str) -> tuple[float, str]:
    x = matrix(names)
    scores = []
    for split in ("random", "scaffold"):
        probability = classifier_oof(model, x, FOLDS[split])
        scores += [roc_auc_score(LABELS, probability), competition_mse(probability, Y, FOLDS[split])]
    text = (f"{label:34} {model:8} {scores[0]:7.3f} {scores[1]:12,.0f} "
            f"{scores[2]:7.3f} {scores[3]:12,.0f}")
    return scores[0], text


def main() -> None:
    print("— каждый блок отдельно —")
    print(HEADER)
    single = {}
    for name in BLOCKS:
        for model in MODELS:
            auc, text = row(name, (name,), model)
            print(text)
            single[(name, model)] = auc

    print("\n— жадное наращивание, порядок по AUC на случайном разбиении —")
    print(HEADER)
    for model in MODELS:
        order = sorted(BLOCKS, key=lambda name: -single[(name, model)])
        chosen: list[str] = []
        best = -np.inf
        for candidate in order:
            auc, text = row("+".join(chosen + [candidate]), chosen + [candidate], model)
            print(text)
            if auc > best:
                best, chosen = auc, chosen + [candidate]
        print(f"{model}: выбрано {'+'.join(chosen)}, AUC {best:.3f}\n")


if __name__ == "__main__":
    main()
