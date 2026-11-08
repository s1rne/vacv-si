"""Frozen CheMeleon embeddings; no supervised fitting or external SMILES upload."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from vacv.dataset import load

CHECKPOINT_MD5 = "6a80b54fdb7de37ef0374d302f01e8ce"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(smiles, checkpoint, batch_size=32):
    import torch
    from chemprop import featurizers, nn
    from chemprop.data import BatchMolGraph

    if hashlib.md5(Path(checkpoint).read_bytes()).hexdigest() != CHECKPOINT_MD5:
        raise ValueError("CheMeleon checkpoint checksum mismatch")
    torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = nn.BondMessagePassing(**state["hyper_parameters"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    aggregate = nn.MeanAggregation()
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    valid = [i for i, mol in enumerate(molecules) if mol is not None and mol.GetNumAtoms()]
    features = np.zeros((len(smiles), model.output_dim), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(valid), batch_size):
            indices = valid[start:start + batch_size]
            graph = BatchMolGraph([featurizer(molecules[i]) for i in indices])
            features[indices] = aggregate(model(graph), graph.batch).numpy()
            print(f"Encoded {min(start + batch_size, len(valid))}/{len(valid)}", flush=True)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite embeddings")
    return features, sorted(set(range(len(smiles))) - set(valid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("data/raw/models/chemeleon_mp.pt"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/models/chemeleon_features.npz"))
    args = parser.parse_args()
    data = load()
    smiles = list(dict.fromkeys([*data.train.SMILES, *data.test.SMILES]))
    features, invalid = encode(smiles, args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, smiles=np.asarray(smiles), embeddings=features)
    manifest = {
        "source": "https://zenodo.org/records/15460715", "license": "MIT",
        "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_md5": CHECKPOINT_MD5,
        "embedding_sha256": sha256(args.out), "shape": list(features.shape),
        "invalid_indices": invalid, "frozen": True,
        "batch_size": 32, "torch_threads": 4, "aggregation": "mean",
        "inputs_sha256": {name: sha256(Path("data/raw/kaggle") / name)
                          for name in ("train.csv", "test.csv")},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "chemprop", "rdkit", "numpy")},
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
