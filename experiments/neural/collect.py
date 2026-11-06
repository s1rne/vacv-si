"""Сводная таблица всех нейросетевых прогонов и их польза в ансамбле.

Читает сохранённые out-of-fold вероятности, печатает сравнение с бустингом и для каждой
модели — корреляцию с рабочим ансамблем и MSE смеси. Ничего не обучает.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vacv.baseline import baselines
from vacv.neural import store
from vacv.neural.ensemble import HEADER as BLEND_HEADER, blend
from vacv.protocol import HEADER, Result, score, task

CHEMPROP = Path("data/interim/neural")


def chemprop_results(job) -> list[Result]:
    out = []
    for path in sorted(CHEMPROP.glob("chemprop.*.npz")):
        name = path.stem.split(".", 1)[1]
        stored = np.load(path, allow_pickle=False)
        meta = {item["split"]: item["seconds"] for item in json.loads(str(stored["meta"]))}
        for split in ("random", "scaffold"):
            if split not in stored.files:
                continue
            probability = stored[split]
            auc, mse = score(job.labels, job.y, probability, job.active_mean)
            out.append(Result(name, split, auc, mse, meta.get(split, float("nan")), probability))
    return out


def main() -> None:
    job = task()
    reference = {split: baselines(job, split) for split in ("random", "scaffold")}
    neural = [row for group in store.groups() for row in store.load(group)]
    neural += chemprop_results(job)

    print(HEADER)
    for split in ("random", "scaffold"):
        for result in reference[split]:
            print(result.row())
        for result in sorted((r for r in neural if r.split == split), key=lambda r: -r.auc):
            print(result.row())
        print()

    print(BLEND_HEADER)
    for split in ("random", "scaffold"):
        base = reference[split][1]
        for result in sorted((r for r in neural if r.split == split), key=lambda r: r.mse)[:12]:
            print(blend(job, base, result).row())
        print()


if __name__ == "__main__":
    main()
