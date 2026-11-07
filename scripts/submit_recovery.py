"""Export the clean calibrated hybrid, preserving competition template order."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from vacv.dataset import load, submission
from vacv.recovery_model import RecoveryModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    components_path = args.out.with_name(args.out.stem + "-components.csv")
    metadata_path = args.out.with_suffix(".json")
    if any(path.exists() for path in (args.out, components_path, metadata_path)):
        parser.error("Choose new output paths; existing artifacts will not be overwritten")
    data = load()
    model = RecoveryModel().fit(data.train, data.y)
    components = model.predict_components(data.test)
    result = submission(data.test["index"].to_numpy(), components["prediction"])
    template = pd.read_csv("data/raw/kaggle/sample_submit.csv")
    if (list(result) != list(template) or template["index"].duplicated().any()
            or set(result["index"]) != set(template["index"])):
        raise ValueError("Submission must match the competition template")
    result = result.set_index("index").loc[template["index"]].reset_index()
    component_frame = pd.DataFrame({"index": data.test["index"].to_numpy(), **components})
    component_frame = component_frame.set_index("index").loc[template["index"]].reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    component_frame.to_csv(components_path, index=False)
    paths = ["vacv/recovery_model.py", "vacv/clean_features.py", "vacv/inference.py",
             "vacv/kernel.py", "vacv/pipeline.py", "data/processed/competition_presence.csv",
             "data/raw/kaggle/train.csv", "data/raw/kaggle/test.csv"]
    metadata = {"profile": "clean-hybrid7-temperature1.2", "rows": len(result),
                "submitted_to_kaggle": False, "exponent": model.exponent,
                "active_mean": model.active_mean_, "lookup_substitution": False,
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "components_sha256": hashlib.sha256(components_path.read_bytes()).hexdigest(),
                "input_code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
