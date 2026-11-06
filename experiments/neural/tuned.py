"""Файнтюнинг на лучших настройках, оба разбиения, с аугментацией и переносом.

Запускается после `tuning.py`: настройки берутся оттуда. Четыре строки на модель —
без аугментации, с размножением случайных SMILES, с усреднением на предсказании,
и с предобучением на внешнем наборе VACV.
"""

from __future__ import annotations

import sys

from vacv.neural import store
from vacv.neural.finetune import Config
from vacv.neural.runner import classify
from vacv.protocol import HEADER, task

BEST = {"chemberta-77m-mtr": (3e-5, 25), "molformer-xl": (3e-5, 25)}


def plan(model: str, rate: float, epochs: int) -> list[tuple[Config, bool]]:
    base = dict(model=model, learning_rate=rate, epochs=epochs)
    return [
        (Config(**base, label=f"{model} файнтюн"), False),
        (Config(**base, augment=5, label=f"{model} файнтюн + aug5"), False),
        (Config(**base, augment=5, tta=5, label=f"{model} файнтюн + aug5 + tta5"), False),
        (Config(**base, label=f"{model} перенос с внешних 1308",
                extra={"pretrain_epochs": 10, "pretrain_lr": 5e-5}), True),
        (Config(**base, freeze_backbone=True, label=f"{model} только голова (тело заморожено)"), False),
    ]


def main() -> None:
    job = task()
    overrides = dict(BEST)
    if len(sys.argv) > 2:
        overrides = {sys.argv[1]: (float(sys.argv[2]), int(sys.argv[3]))}
    results = []
    print(HEADER)
    for model, (rate, epochs) in overrides.items():
        for config, pretrain in plan(model, rate, epochs):
            for split in ("random", "scaffold"):
                result = classify(job, split, config, pretrain=pretrain)
                results.append(result)
                print(result.row(), flush=True)
    store.save("tuned", results)


if __name__ == "__main__":
    main()
