"""Сводная таблица всех прогонов и польза каждой модели в ансамбле.

Читает сохранённые out-of-fold вероятности, ничего не обучает.

Таблица смешивания печатается дважды. Сначала по всем моделям: там наверху оказываются
те, что читают SMILES как в файле, и их выигрыш — манера записи строки (см. раздел 2
в `docs/neural.md`). Потом по моделям, которым эта подсказка недоступна: по каноническим
SMILES и по графу. Вторая таблица и отвечает на вопрос, нужна ли нейросеть в ансамбле.
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
# Графовым сетям и обычным признакам манера записи строки недоступна по построению:
# отпечатки и граф считаются по разобранной молекуле, а не по строке.
GRAPH_PREFIXES = ("GIN", "ATTENTIVEFP", "dmpnn")
CLASSIC = {"morgan counts + linear", "MACCS + linear", "дескрипторы RDKit + linear",
           "morgan + MACCS + linear"}


def leak_free(name: str) -> bool:
    """Видит ли модель запись SMILES из файла.

    Пометка «канон» означает, что строки переписаны канонически. Приклеенные отпечатки
    («+ morgan») сами по себе ничего не исправляют: эмбеддинг рядом с ними всё равно
    посчитан по строке из файла.
    """
    return ("канон" in name or name in CLASSIC
            or name.startswith(GRAPH_PREFIXES))


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
    # «[как в файле]» из controls.py — те же прогоны, что в frozen.py, под другим именем.
    neural = [row for row in neural if "[как в файле]" not in row.model]
    neural += chemprop_results(job)

    print(HEADER)
    for split in ("random", "scaffold"):
        for result in reference[split]:
            print(result.row())
        for result in sorted((r for r in neural if r.split == split), key=lambda r: -r.auc):
            print(result.row())
        print()

    for title, keep in (("все модели", lambda name: True),
                        ("только без подсказки от записи SMILES", leak_free)):
        print(f"# смешивание с ансамблем пайплайна: {title}")
        print(BLEND_HEADER)
        for split in ("random", "scaffold"):
            base = next(r for r in reference[split] if r.model.startswith("GBDT blend ("))
            chosen = sorted((r for r in neural if r.split == split and keep(r.model)),
                            key=lambda r: r.mse)[:10]
            for result in chosen:
                print(blend(job, base, result).row())
            print()


if __name__ == "__main__":
    main()
