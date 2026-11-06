"""Хранилище результатов прогонов.

Каждый прогон — одна строка таблицы и один вектор out-of-fold вероятностей. Строки
складываются в `data/interim/neural/<группа>.npz`, чтобы сводную таблицу и смешивание
можно было пересобрать, не обучая заново: файнтюнинг пяти фолдов занимает минуты,
а таблицу хочется переверстать мгновенно.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..protocol import Result

ROOT = Path("data/interim/neural")


def save(group: str, results: list[Result]) -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{group}.npz"
    payload = {f"p{index}": result.probability for index, result in enumerate(results)}
    payload["meta"] = np.array(json.dumps(
        [{"model": r.model, "split": r.split, "auc": r.auc, "mse": r.mse, "seconds": r.seconds}
         for r in results]))
    np.savez(path, **payload)
    return path


def load(group: str) -> list[Result]:
    path = ROOT / f"{group}.npz"
    if not path.exists():
        return []
    stored = np.load(path, allow_pickle=False)
    meta = json.loads(str(stored["meta"]))
    return [Result(item["model"], item["split"], item["auc"], item["mse"], item["seconds"],
                   stored[f"p{index}"]) for index, item in enumerate(meta)]


def groups() -> list[str]:
    return sorted(path.stem for path in ROOT.glob("*.npz")) if ROOT.exists() else []
