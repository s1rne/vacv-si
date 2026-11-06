"""Обучение на всей выборке и файл ответа для Kaggle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vacv.dataset import load, submission
from vacv.pipeline import SelectivityModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--seeds", type=int, default=7)
    args = parser.parse_args()

    data = load()
    predictions = [SelectivityModel(seed).fit(data.train, data.y).predict(data.test) for seed in range(args.seeds)]
    prediction = np.mean(predictions, axis=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission(data.test["index"].to_numpy(), prediction).to_csv(args.out, index=False)
    print(f"{args.out}: {len(prediction)} строк, среднее {prediction.mean():.1f}, медиана {np.median(prediction):.1f}")


if __name__ == "__main__":
    main()
