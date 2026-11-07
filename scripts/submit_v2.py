"""Export official-v2 regression predictions with input/code fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from vacv.dataset import load, submission
from vacv.regression_v2 import RegressionV2, PROFILES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_v2"))
    parser.add_argument("--profile", choices=PROFILES, default="et-blend-log")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".json").exists():
        parser.error("Choose unused output paths")
    data = load(args.data_dir, require_v2=True)
    prediction = RegressionV2(args.profile).fit(data.train, data.y).predict(data.test)
    result = submission(data.test["index"].to_numpy(), prediction)
    template = pd.read_csv(args.data_dir / "sample_submit.csv")
    if (list(result) != list(template) or template["index"].duplicated().any()
            or set(result["index"]) != set(template["index"])):
        raise ValueError("Submission does not match template")
    result = result.set_index("index").loc[template["index"]].reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    paths = [args.data_dir / name for name in ("train_v2.csv", "test.csv", "sample_submit.csv")]
    paths += [Path(p) for p in ("vacv/regression_v2.py", "vacv/dataset.py", "vacv/pipeline.py", "vacv/kernel.py")]
    metadata = {"target_version": "official-train_v2", "profile": args.profile,
                "rows": len(result), "submitted_to_kaggle": False,
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "input_code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
