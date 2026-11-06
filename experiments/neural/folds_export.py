"""Выгрузка фолдов и метки неактивности для прогонов в отдельном окружении.

Chemprop тянет свой lightning и descriptastorus и ставится в отдельное окружение, где
пакета `vacv` нет. Чтобы разбиения совпадали до индекса, они выгружаются сюда, а скрипт
Chemprop читает готовый файл и ничего не делит сам.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vacv.protocol import task

OUT = Path("data/interim/neural/folds.npz")


def main() -> None:
    job = task()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"smiles": np.array(list(job.frame["SMILES"])),
               "labels": job.labels,
               "y": job.y}
    for split, folds in job.folds.items():
        for index, fold in enumerate(folds):
            payload[f"{split}.{index}"] = fold
    np.savez(OUT, **payload)
    print(f"{OUT}: {len(job.y)} молекул, {job.labels.sum()} неактивных")


if __name__ == "__main__":
    main()
