"""D-MPNN (Chemprop 2.3.1) на тех же фолдах.

Запускается в отдельном окружении, где стоит только Chemprop: он тянет lightning
и descriptastorus, которые конфликтуют с transformers. Поэтому скрипт не импортирует
`vacv`, а читает фолды из `data/interim/neural/folds.npz`, собранного `folds_export.py`,
и пишет out-of-fold вероятности в `data/interim/neural/chemprop.<конфигурация>.npz`.

Внутри обучающей части фолда десятая доля отводится на раннюю остановку: без валидации
Chemprop берёт последнюю эпоху, и 50 эпох на 230 молекулах её переобучают.

    uv run --project ~/hack/neural-vacv-chemprop python experiments/neural/chemprop_run.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = Path("data/interim/neural/folds.npz")
WORK = Path("data/interim/neural/chemprop")
EPOCHS = 50
PATIENCE = 15

CONFIGS = {
    "dmpnn": [],
    "dmpnn-rdkit": ["--molecule-featurizers", "v1_rdkit_2d_normalized"],
    "dmpnn-rdkit-morgan": ["--molecule-featurizers", "v1_rdkit_2d_normalized", "morgan_count"],
}


def run(command: list[str]) -> None:
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        sys.stderr.write(finished.stdout[-4000:] + finished.stderr[-4000:])
        raise SystemExit(f"не выполнилось: {' '.join(command)}")


def fold_run(directory: Path, frame: pd.DataFrame, train_index: np.ndarray,
             validation_index: np.ndarray, options: list[str], seed: int) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    marks = pd.Series("test", index=frame.index)
    rng = np.random.default_rng(seed)
    shuffled = train_index.copy()
    rng.shuffle(shuffled)
    cut = max(1, int(0.1 * len(shuffled)))
    marks.iloc[shuffled[cut:]] = "train"
    marks.iloc[shuffled[:cut]] = "val"
    table = frame.assign(split=marks)
    data_path = directory / "data.csv"
    table.to_csv(data_path, index=False)

    run(["chemprop", "train", "-i", str(data_path), "--splits-column", "split",
         "-s", "SMILES", "--target-columns", "inactive", "-t", "classification",
         "--epochs", str(EPOCHS), "--patience", str(PATIENCE),
         "--accelerator", "gpu", "--devices", "1", "--pytorch-seed", str(seed),
         "-o", str(directory / "out"), "--quiet", *options])

    checkpoints = sorted((directory / "out").rglob("best.pt"))
    if not checkpoints:
        checkpoints = sorted((directory / "out").rglob("*.ckpt"))
    holdout = directory / "holdout.csv"
    frame.iloc[validation_index].to_csv(holdout, index=False)
    predictions = directory / "preds.csv"
    run(["chemprop", "predict", "-i", str(holdout), "-s", "SMILES",
         "--model-path", str(checkpoints[-1]), "--preds-path", str(predictions),
         "--accelerator", "gpu", "--devices", "1", *options])
    result = pd.read_csv(predictions)
    column = [name for name in result.columns if name != "SMILES"][-1]
    return result[column].to_numpy(dtype=float)


def main() -> None:
    stored = np.load(FOLDS, allow_pickle=False)
    frame = pd.DataFrame({"SMILES": stored["smiles"], "inactive": stored["labels"].astype(int)})
    folds = {split: [stored[f"{split}.{i}"] for i in range(5)] for split in ("random", "scaffold")}
    WORK.mkdir(parents=True, exist_ok=True)

    for name, options in CONFIGS.items():
        meta = []
        payload = {}
        for split, parts in folds.items():
            probability = np.zeros(len(frame))
            started = time.perf_counter()
            for fold in range(len(parts)):
                validation_index = parts[fold]
                train_index = np.concatenate([p for i, p in enumerate(parts) if i != fold])
                directory = WORK / f"{name}.{split}.{fold}"
                shutil.rmtree(directory, ignore_errors=True)
                probability[validation_index] = fold_run(directory, frame, np.sort(train_index),
                                                         validation_index, options, seed=fold)
                shutil.rmtree(directory, ignore_errors=True)
            payload[f"{split}"] = probability
            meta.append({"split": split, "seconds": time.perf_counter() - started})
            print(f"{name} {split}: готово за {meta[-1]['seconds']:.0f} с", flush=True)
        payload["meta"] = np.array(json.dumps(meta))
        np.savez(Path("data/interim/neural") / f"chemprop.{name}.npz", **payload)


if __name__ == "__main__":
    main()
