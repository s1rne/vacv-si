import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from vacv.checkpoint import combine, load_embeddings


def test_checkpoint_blends_probability_before_single_soft_lookup():
    frame = pd.DataFrame({"literature_si": [10., np.nan]})
    p, raw, prediction = combine(np.array([.2, .8]), np.array([.8, .2]), 500., frame)
    assert np.allclose(p, [.35, .65])
    assert np.allclose(raw, [3825., 6675.])
    assert np.allclose(prediction, [3506.5, 6675.])


def test_checkpoint_rejects_incorrect_component_alignment():
    with pytest.raises(ValueError):
        combine(np.array([.2]), np.array([.8, .2]), 5., pd.DataFrame({"literature_si": [1.]}))


def test_checkpoint_rejects_out_of_range_component_even_if_average_is_valid():
    with pytest.raises(ValueError):
        combine(np.array([-.1]), np.array([.9]), 5., pd.DataFrame({"literature_si": [1.]}))


def test_embedding_loader_checks_manifest_and_duplicate_smiles(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, smiles=np.array(["CCO", "CCO"]), embeddings=np.ones((2, 2048)))
    path.with_suffix(".json").write_text(json.dumps({
        "embedding_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match="unique SMILES"):
        load_embeddings(path)
    path.with_suffix(".json").write_text(json.dumps({"embedding_sha256": "incorrect"}))
    with pytest.raises(ValueError, match="checksum"):
        load_embeddings(path)
