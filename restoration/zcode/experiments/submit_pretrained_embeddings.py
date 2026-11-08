"""Export the frozen 25% CheTabICL / 75% hybrid7 candidate without uploading."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.pretrained_embeddings import predict_head
from vacv.dataset import load, submission
from vacv.inference import predict_ensemble
from vacv.literature import attach
from vacv.pipeline import INACTIVE, SENTINEL
from vacv.postprocess import conditional_lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/raw/models/chemeleon_features.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/raw/models/tabicl-classifier-v2-20260212.ckpt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/submission-chemeleon-soft.csv"))
    args = parser.parse_args()
    manifest = json.loads(args.features.with_suffix(".json").read_text())
    if hashlib.sha256(args.features.read_bytes()).hexdigest() != manifest["embedding_sha256"]:
        raise ValueError("Embedding cache checksum mismatch")
    expected = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != expected:
        raise ValueError("TabICL checkpoint checksum mismatch")
    with np.load(args.features, allow_pickle=False) as archive:
        embeddings = dict(zip(archive["smiles"], archive["embeddings"], strict=True))
    data = load()
    train, test = attach(data.train), attach(data.test)
    with threadpool_limits(limits=4):
        base_p, base_raw, _ = predict_ensemble(train, data.y, test, 7, "hybrid")
        head_p = predict_head(train, data.y, test, np.stack([embeddings[s] for s in train.SMILES]),
                              np.stack([embeddings[s] for s in test.SMILES]), "che-tabicl", args.checkpoint)
    active_mean = data.y[data.y != INACTIVE].mean()
    probability = .75 * base_p + .25 * head_p
    head_raw = head_p * SENTINEL + (1-head_p) * active_mean
    raw = .75 * base_raw + .25 * head_raw
    prediction = conditional_lookup(raw, probability, test)
    result = submission(data.test["index"].to_numpy(), prediction)
    template = pd.read_csv("data/raw/kaggle/sample_submit.csv")
    if (list(template.columns) != list(result.columns) or template["index"].duplicated().any()
            or set(template["index"]) != set(result["index"])):
        raise ValueError("Submission indices/columns do not match sample_submit.csv")
    result = result.set_index("index").loc[template["index"]].reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    components = pd.DataFrame({"index": data.test["index"], "hybrid_probability": base_p,
                               "head_probability": head_p, "probability": probability,
                               "model_prediction": raw, "prediction": prediction})
    component_path = args.out.with_name(args.out.stem + "-components.csv")
    components.to_csv(component_path, index=False)
    metadata = {"profile": "0.75*hybrid7+0.25*CheMeleonPCA32TabICL2-soft", "rows": len(result),
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "components_sha256": hashlib.sha256(component_path.read_bytes()).hexdigest(),
                "embedding_sha256": manifest["embedding_sha256"], "tabicl_sha256": expected}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
