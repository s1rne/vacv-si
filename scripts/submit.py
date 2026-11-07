"""Обучение на всей выборке и файл ответа для Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vacv.dataset import load, submission
from vacv.literature import attach, to_zero_inactive
from vacv.inference import PROFILES, predict_ensemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--profile", choices=PROFILES, default="hybrid")
    args = parser.parse_args()

    data = load()
    train, test = attach(data.train), attach(data.test)
    probability, _, prediction = predict_ensemble(train, data.y, test, args.seeds, args.profile)
    prediction = to_zero_inactive(prediction, probability, test)
    print(f"значений из литературы: {int(test['matched'].sum())} из {len(test)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = submission(data.test["index"].to_numpy(), prediction)
    template_path = Path("data/raw/kaggle/sample_submit.csv")
    if template_path.exists():
        template = pd.read_csv(template_path)
        if (list(template.columns) != list(result.columns)
                or template["index"].duplicated().any()
                or set(template["index"]) != set(result["index"])):
            raise ValueError("Submission indices/columns do not match sample_submit.csv")
        result = result.set_index("index").loc[template["index"]].reset_index()
    result.to_csv(args.out, index=False)
    metadata = {"profile": args.profile, "seeds": args.seeds, "rows": len(result),
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(prediction)} строк, среднее {prediction.mean():.1f}, медиана {np.median(prediction):.1f}")


if __name__ == "__main__":
    main()
