"""Замороженные эмбеддинги химических языковых моделей против отпечатков.

Самая дешёвая проверка: предобученная модель считает по одному вектору на молекулу, сверху
обучается голова. Если представление сильнее отпечатков, это видно уже здесь и дальше можно
не идти. Головы: линейный зонд с вложенным выбором C, MLP, и ExtraTrees — лес нужен, чтобы
отделить слабое представление от слабой головы.

Отдельно считается конкатенация эмбеддинга с отпечатками Моргана: если эмбеддинг добавляет
к отпечаткам хоть что-то, конкатенация обойдёт оба блока по отдельности.
"""

from __future__ import annotations

import numpy as np

from vacv.baseline import baselines
from vacv.neural import embed, store
from vacv.neural.probe import active_regression, probe
from vacv.pipeline import structure_matrix
from vacv.protocol import HEADER, score, task

POOLING = "mean"


def main() -> None:
    job = task()
    smiles = list(job.frame["SMILES"])
    morgan = structure_matrix(job.frame)
    results = []

    print(HEADER)
    for split in ("random", "scaffold"):
        for result in baselines(job, split):
            print(result.row())

    cached = {}
    for name in embed.MODELS:
        embedding = embed.load(name, smiles)
        cached[name] = embedding
        print(f"# {name}: dim {embedding.mean.shape[1]}, {embedding.seconds:.1f} с на 320 молекул")

    for split in ("random", "scaffold"):
        for name, embedding in cached.items():
            matrix = embedding.variant(POOLING)
            for head in ("linear", "mlp", "extra"):
                result = probe(job, split, matrix, name, head, extra_seconds=embedding.seconds)
                results.append(result)
                print(result.row())
            joined = np.hstack([matrix, morgan])
            result = probe(job, split, joined, f"{name} + morgan", "linear",
                           extra_seconds=embedding.seconds)
            results.append(result)
            print(result.row())

    best = max((r for r in results if r.split == "random"), key=lambda r: r.auc)
    print(f"\n# лучший на случайном разбиении: {best.model}")
    for split in ("random", "scaffold"):
        name = best.model.split(" + ")[0]
        matrix = cached[name].variant(POOLING)
        regression = active_regression(job, split, matrix)
        probability = next(r.probability for r in results if r.model == best.model and r.split == split)
        constant = score(job.labels, job.y, probability, job.active_mean)
        learned = score(job.labels, job.y, probability, regression)
        print(f"  {split:9} SI активных константой {constant[1]:13,.0f}"
              f"  регрессией на эмбеддинге {learned[1]:13,.0f}")

    for split in ("random", "scaffold"):
        for name, embedding in cached.items():
            for kind in ("cls", "pooler"):
                if kind == "pooler" and embedding.pooler is None:
                    continue
                result = probe(job, split, embedding.variant(kind), f"{name}[{kind}]", "linear",
                               extra_seconds=embedding.seconds)
                results.append(result)
                print(result.row())

    store.save("frozen", results)


if __name__ == "__main__":
    main()
