"""Rebuild and export the complete intermediate best ensemble locally."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from vacv.checkpoint import BestCheckpointModel, load_embeddings
from vacv.dataset import load, submission
from vacv.literature import attach


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/raw/models/chemeleon_features.npz"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/raw/models/tabicl-classifier-v2-20260212.ckpt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/checkpoint/submission.csv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    component_path = args.out.with_name(args.out.stem + "-components.csv")
    metadata_path = args.out.with_suffix(".json")
    if not args.overwrite and any(p.exists() for p in (args.out, component_path, metadata_path)):
        parser.error("Output already exists: choose a new path or pass --overwrite")
    embeddings, manifest = load_embeddings(args.features)
    data = load()
    train, test = attach(data.train), attach(data.test)
    model = BestCheckpointModel(embeddings, args.checkpoint).fit(train, data.y)
    components = model.predict_components(test)
    result = submission(data.test["index"].to_numpy(), components["prediction"])
    template = pd.read_csv("data/raw/kaggle/sample_submit.csv")
    if (list(template) != list(result) or template["index"].duplicated().any()
            or set(template["index"]) != set(result["index"])):
        raise ValueError("Submission does not match the competition template")
    result = result.set_index("index").loc[template["index"]].reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    pd.DataFrame({"index": data.test["index"].to_numpy(), **components}).to_csv(component_path, index=False)
    code_files = ["vacv/checkpoint.py", "vacv/graph_correction.py", "vacv/postprocess.py",
                  "vacv/inference.py", "vacv/pipeline.py", "experiments/pretrained_embeddings.py"]
    metadata = {"profile": "0.75*gated_graph_hybrid7+0.25*CheMeleonPCA32TabICL2-soft",
                "rows": len(result), "stage": "intermediate checkpoint; not submitted to Kaggle",
                "transductive": True, "embedding_sha256": manifest["embedding_sha256"],
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "components_sha256": hashlib.sha256(component_path.read_bytes()).hexdigest(),
                "code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code_files}}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
