"""Отсчёт для сравнения: текущее решение на отпечатках и бустинге.

Два отсчёта, потому что нейросетевые модели читают только SMILES:

- `structure` — ExtraTrees на счётных отпечатках Моргана и ключах MACCS. Честное сравнение
  с любой моделью, которая тоже видит только строение;
- `blend` — весь рабочий пайплайн (`SelectivityModel`), включая штамм, культуру,
  обозначение соединения и долю неактивных в группе. Это то, что реально отправляется
  на Kaggle, и верхняя планка.

Вероятности считаются out-of-fold на тех же фолдах, что и у нейросетей, и кладутся
в `data/interim/oof/`, чтобы смешивание и корреляции считались без повторного обучения.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from .literature import attach
from .pipeline import SelectivityModel, structure_matrix
from .protocol import RANDOM_SEED, Result, Task

CACHE = Path("data/interim/oof")


def forest(seed: int = RANDOM_SEED) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)


def structure_oof(job: Task, split: str) -> Result:
    matrix = structure_matrix(job.frame)
    probability = np.zeros(len(job.y))
    started = time.perf_counter()
    for train_index, validation_index in job.splits(split):
        probability[validation_index] = forest().fit(matrix[train_index], job.labels[train_index]) \
            .predict_proba(matrix[validation_index])[:, 1]
    elapsed = time.perf_counter() - started
    auc, mse = _score(job, probability)
    return Result("GBDT structure (ExtraTrees)", split, auc, mse, elapsed, probability)


def blend_oof(job: Task, split: str) -> Result:
    return _blend(job, split, attach(job.frame), "GBDT blend (пайплайн)")


def plain_blend_oof(job: Task, split: str) -> Result:
    """Тот же ансамбль без признаков присутствия в открытых базах.

    Признаки присутствия включают литературные IC50 и CC50 самой молекулы. На тестовой
    выборке это законно — справочник доступен и там, — но при перекрёстной проверке они
    подсказывают ответ и завышают отсчёт. Нейросети таких подсказок не дают, поэтому
    сравнение с ними честнее вести по этой строке.
    """
    return _blend(job, split, job.frame, "GBDT blend без открытых данных")


def _blend(job: Task, split: str, frame, name: str) -> Result:
    probability = np.zeros(len(job.y))
    started = time.perf_counter()
    for train_index, validation_index in job.splits(split):
        model = SelectivityModel().fit(frame.iloc[train_index], job.y[train_index])
        probability[validation_index] = model.probability(frame.iloc[validation_index])
    elapsed = time.perf_counter() - started
    auc, mse = _score(job, probability)
    return Result(name, split, auc, mse, elapsed, probability)


def _score(job: Task, probability: np.ndarray) -> tuple[float, float]:
    from .protocol import score

    return score(job.labels, job.y, probability, job.active_mean)


def cached(job: Task, name: str, split: str, build) -> Result:
    """Результат из `data/interim/oof/<name>.<split>.npz` или посчитанный и записанный туда."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.{split}.npz"
    if path.exists():
        stored = np.load(path, allow_pickle=False)
        auc, mse = _score(job, stored["probability"])
        return Result(str(stored["model"]), split, auc, mse, float(stored["seconds"]), stored["probability"])
    result = build(job, split)
    np.savez(path, probability=result.probability, seconds=result.seconds, model=result.model)
    return result


def baselines(job: Task, split: str) -> list[Result]:
    return [cached(job, "gbdt-structure", split, structure_oof),
            cached(job, "gbdt-plain", split, plain_blend_oof),
            cached(job, "gbdt-blend", split, blend_oof)]


def main() -> None:
    from .protocol import HEADER, task

    job = task()
    print(HEADER)
    for split in ("random", "scaffold"):
        for result in baselines(job, split):
            print(result.row())


if __name__ == "__main__":
    main()
