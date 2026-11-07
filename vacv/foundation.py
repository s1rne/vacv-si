"""Offline TabICL candidate on compact, permitted molecular/context features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors

from .features import parse
from .pipeline import INACTIVE, SENTINEL, open_data_matrix, title_matrix

DESCRIPTOR_NAMES = (
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount",
    "NumHeteroatoms", "NumAliphaticRings", "LabuteASA", "BertzCT", "BalabanJ", "MaxPartialCharge",
)
MODEL_PATH = Path("data/raw/models/tabicl-classifier-v2-20260212.ckpt")


def compact_features(frame):
    rows = []
    for smiles in frame["SMILES"]:
        mol = parse(smiles)
        values = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan) if mol is not None else {}
        rows.append([values.get(name, np.nan) for name in DESCRIPTOR_NAMES])
    matrix = np.hstack([rows, title_matrix(frame["Title"]), open_data_matrix(frame)])
    result = pd.DataFrame(matrix, columns=[f"numeric_{i}" for i in range(matrix.shape[1])])
    result = result.replace([np.inf, -np.inf], np.nan)
    for column in ("strain", "Culture"):
        result[column] = frame[column].fillna("").astype(str).to_numpy()
    return result


class FoundationSelectivityModel:
    def __init__(self, n_estimators=2, model_path=MODEL_PATH):
        self.n_estimators = n_estimators
        self.model_path = model_path

    def fit_predict(self, train, y, validation, unlabeled=None):
        from tabicl import TabICLClassifier
        import torch

        torch.set_num_threads(4)
        y = np.asarray(y)
        labels = (y == INACTIVE).astype(int)
        if len(np.unique(labels)) < 2:
            probability = np.full(len(validation), labels[0], dtype=float)
        else:
            model = TabICLClassifier(
                n_estimators=self.n_estimators, batch_size=1, device="cpu", use_amp=False,
                n_jobs=4, model_path=self.model_path, allow_auto_download=False, random_state=0,
            )
            model.fit(compact_features(train), labels)
            probability = model.predict_proba(compact_features(validation))[:, 1]
        active_mean = float(y[labels == 0].mean()) if (labels == 0).any() else 0.0
        return probability, probability * SENTINEL + (1 - probability) * active_mean
