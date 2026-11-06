"""Перебор настроек файнтюнинга на случайном разбиении.

Перебор идёт по той же выборке, на которой потом печатается результат, поэтому числа
оптимистичны. Это сделано нарочно: нейросети даётся преимущество, которого нет у бустинга
с фиксированными настройками. Если она проигрывает и так, вывод устойчив.
"""

from __future__ import annotations

import itertools

from vacv.neural import store
from vacv.neural.finetune import Config
from vacv.neural.runner import classify
from vacv.protocol import HEADER, task

MODELS = ("chemberta-77m-mtr", "molformer-xl")
RATES = (1e-5, 3e-5, 1e-4)
EPOCHS = (10, 25, 50)


def main() -> None:
    import sys

    group = "tuning"
    rates, epochs_grid = RATES, EPOCHS
    if len(sys.argv) > 1:
        group, rates, epochs_grid = sys.argv[1], \
            tuple(float(value) for value in sys.argv[2].split(",")), \
            tuple(int(value) for value in sys.argv[3].split(","))

    job = task()
    results = []
    print(HEADER)
    for model, rate, epochs in itertools.product(MODELS, rates, epochs_grid):
        config = Config(model=model, learning_rate=rate, epochs=epochs)
        result = classify(job, "random", config)
        results.append(result)
        print(result.row(), flush=True)
        store.save(group, results)
    store.save(group, results)
    best = max(results, key=lambda r: r.auc)
    print(f"\n# лучшая настройка по AUC: {best.model}")
    best_mse = min(results, key=lambda r: r.mse)
    print(f"# лучшая настройка по MSE: {best_mse.model}")


if __name__ == "__main__":
    main()
