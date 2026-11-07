"""Обучение на всей выборке и файл ответа для Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vacv.dataset import load, submission
from vacv.clean_features import attach_clean
from vacv.literature import attach
from vacv.inference import PROFILES, predict_ensemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--profile", choices=PROFILES, default="hybrid")
    parser.add_argument("--lookup-policy", choices=["clean", "recovered"], default="clean")
    args = parser.parse_args()

    data = load()
    attach_features = attach_clean if args.lookup_policy == "clean" else attach
    train, test = attach_features(data.train), attach_features(data.test)
    _, _, prediction = predict_ensemble(train, data.y, test, args.seeds, args.profile)
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
                "lookup_policy": args.lookup_policy,
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(prediction)} строк, среднее {prediction.mean():.1f}, медиана {np.median(prediction):.1f}")


if __name__ == "__main__":
    main()
