"""Файнтюнинг на лучших настройках: оба разбиения, аугментация, перенос, канонизация.

Запускается после `tuning.py`, настройки берутся оттуда. Что проверяется каждой строкой:

- `файнтюн` — обычное дообучение всей модели;
- `+ aug5` — каждая молекула размножена пятью случайными записями SMILES;
- `+ tta5` — плюс усреднение предсказания по пяти записям;
- `перенос с внешних 1308` — тело сперва дообучено на внешнем наборе VACV на log IC50,
  log CC50 и метку «IC50 не достигнута», валидационные соединения оттуда выкинуты;
- `только голова` — тело заморожено, обучается один линейный слой;
- `[канон]` — те же SMILES, переписанные канонически. Разница с той же строкой без
  пометки показывает, сколько давала манера записи, а не молекула.

MoLFormer в 14 раз больше ChemBERTa-77M, поэтому для него берётся короткий план.
"""

from __future__ import annotations

import sys

from vacv.neural import store
from vacv.neural.finetune import Config
from vacv.neural.runner import classify
from vacv.protocol import HEADER, task

BEST = {"chemberta-77m-mtr": (3e-5, 25), "molformer-xl": (3e-5, 25)}
SHORT = {"molformer-xl"}


def plan(model: str, rate: float, epochs: int) -> list[tuple[Config, bool]]:
    base = dict(model=model, learning_rate=rate, epochs=epochs)
    full = [
        (Config(**base, label=f"{model} файнтюн"), False),
        (Config(**base, augment=5, label=f"{model} файнтюн + aug5"), False),
        (Config(**base, augment=5, tta=5, label=f"{model} файнтюн + aug5 + tta5"), False),
        (Config(**base, label=f"{model} перенос с внешних 1308",
                extra={"pretrain_epochs": 10, "pretrain_lr": 5e-5}), True),
        (Config(**base, freeze_backbone=True, label=f"{model} только голова (тело заморожено)"), False),
        (Config(**base, canonical=True, label=f"{model} файнтюн [канон]"), False),
        (Config(**base, canonical=True, augment=5, tta=5,
                label=f"{model} файнтюн + aug5 + tta5 [канон]"), False),
        (Config(**base, canonical=True, label=f"{model} перенос с внешних 1308 [канон]",
                extra={"pretrain_epochs": 10, "pretrain_lr": 5e-5}), True),
    ]
    if model in SHORT:
        return [full[0], full[3], full[5], full[7]]
    return full


def main() -> None:
    job = task()
    models = sys.argv[1:] or list(BEST)
    results = []
    print(HEADER)
    for model in models:
        rate, epochs = BEST[model]
        for config, pretrain in plan(model, rate, epochs):
            for split in ("random", "scaffold"):
                result = classify(job, split, config, pretrain=pretrain)
                results.append(result)
                print(result.row(), flush=True)
                store.save("tuned", results)


if __name__ == "__main__":
    main()
