"""Нужна ли нейросеть в ансамбле с текущим бустингом.

Модель бесполезна в ансамбле, если её out-of-fold вероятности почти совпадают с уже
имеющимися: смесь тогда повторяет сильнейшего участника. Поэтому считаются две вещи:
корреляция вероятностей с бустингом и MSE смеси с весами, обратными OOF-MSE участников.

Обратные веса выбираются без подглядывания в контроль. Для сравнения печатается и лучший
вес на сетке — он уже подобран по той же выборке, на которой измеряется, поэтому
оптимистичен и годится только как верхняя граница возможной пользы.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..protocol import Result, Task, score


@dataclass
class Blend:
    model: str
    split: str
    correlation: float
    base_mse: float
    inverse_mse: float
    best_mse: float
    best_weight: float

    def row(self) -> str:
        delta = self.inverse_mse - self.base_mse
        return (f"{self.model:34} {self.split:9} {self.correlation:7.3f} {self.base_mse:13,.0f} "
                f"{self.inverse_mse:13,.0f} {delta:+13,.0f} {self.best_mse:13,.0f} {self.best_weight:6.2f}")


HEADER = (f"{'модель':34} {'разбиение':9} {'corr':>7} {'MSE GBDT':>13} {'MSE обр.':>13} "
          f"{'дельта':>13} {'MSE лучш.':>13} {'вес':>6}")


def blend(job: Task, base: Result, candidate: Result, grid: np.ndarray | None = None) -> Blend:
    grid = np.arange(0.0, 1.01, 0.05) if grid is None else grid
    correlation = float(np.corrcoef(base.probability, candidate.probability)[0, 1])
    weight = (1.0 / candidate.mse) / (1.0 / base.mse + 1.0 / candidate.mse)
    inverse = score(job.labels, job.y,
                    (1 - weight) * base.probability + weight * candidate.probability,
                    job.active_mean)[1]
    scored = [(score(job.labels, job.y, (1 - value) * base.probability + value * candidate.probability,
                     job.active_mean)[1], value) for value in grid]
    best_mse, best_weight = min(scored)
    return Blend(candidate.model, candidate.split, correlation, base.mse, inverse, best_mse, best_weight)
