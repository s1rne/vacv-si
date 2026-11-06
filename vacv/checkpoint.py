"""Fixed intermediate ensemble of the best independently explored components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.pretrained_embeddings import predict_head
from .graph_correction import GraphResidualCorrector, correct_probability, safe_features
from .pipeline import INACTIVE, SENTINEL
from .postprocess import conditional_lookup

TABICL_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"


def load_embeddings(path):
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["embedding_sha256"]:
        raise ValueError("Embedding cache checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        smiles, values = archive["smiles"], archive["embeddings"]
        if (values.ndim != 2 or values.shape != (len(smiles), 2048)
                or not np.isfinite(values).all() or len(np.unique(smiles)) != len(smiles)):
            raise ValueError("Expected unique SMILES with finite 2048-dimensional embeddings")
        return dict(zip(smiles, values, strict=True)), manifest


def combine(probability_graph, probability_chemical, active_mean, frame):
    graph = np.asarray(probability_graph, dtype=float)
    chemical = np.asarray(probability_chemical, dtype=float)
    if graph.shape != (len(frame),) or chemical.shape != graph.shape:
        raise ValueError("Component probabilities must align with query rows")
    if (not np.isfinite(graph).all() or not np.isfinite(chemical).all()
            or (graph < 0).any() or (graph > 1).any()
            or (chemical < 0).any() or (chemical > 1).any()
            or not np.isfinite(active_mean) or active_mean < 0):
        raise ValueError("Finite probabilities in [0, 1] and nonnegative active mean required")
    probability = 0.75 * graph + 0.25 * chemical
    raw = probability * SENTINEL + (1 - probability) * active_mean
    return probability, raw, conditional_lookup(raw, probability, frame)


class BestCheckpointModel:
    """Batch-transductive graph75 / CheMeleon-PCA32-TabICL25, soft lookup once.

    ``query`` must be the complete intended prediction batch: graph geometry
    intentionally depends on the unlabeled pool. No target is read from it.
    """

    def __init__(self, embeddings, checkpoint):
        self.embeddings = embeddings
        self.checkpoint = Path(checkpoint)
        if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != TABICL_SHA256:
            raise ValueError("TabICL checkpoint checksum mismatch")

    def _embedding_matrix(self, frame):
        try:
            return np.stack([self.embeddings[smiles] for smiles in frame["SMILES"]])
        except KeyError as error:
            raise ValueError("Missing embedding: regenerate the frozen feature cache for this dataset") from error

    def fit(self, train, y):
        self.train_ = safe_features(train)
        self.y_ = np.asarray(y, dtype=float).copy()
        if (self.y_.shape != (len(train),) or not np.isfinite(self.y_).all()
                or (self.y_ < 0).any()):
            raise ValueError("Finite aligned nonnegative training targets required")
        self.embedding_train_ = self._embedding_matrix(self.train_)
        with threadpool_limits(limits=4):
            self.graph_ = GraphResidualCorrector(seeds=7).fit(self.train_, self.y_)
        active = self.y_[self.y_ != INACTIVE]
        self.active_mean_ = float(active.mean()) if len(active) else 0.0
        return self

    def predict_components(self, query, unlabeled=None):
        query = safe_features(query)
        with threadpool_limits(limits=4):
            components = self.graph_.components(query, unlabeled)
            graph_probability = correct_probability(
                components["probability"], components["deltas"]["diffusion"], .25,
                uncertainty_gate=True)
            chemical_probability = predict_head(
                self.train_, self.y_, query, self.embedding_train_, self._embedding_matrix(query),
                "che-tabicl", self.checkpoint)
            chemical_probability = np.asarray(chemical_probability, dtype=float)
        probability, raw, prediction = combine(graph_probability, chemical_probability,
                                               self.active_mean_, query)
        return {"hybrid_probability": components["probability"],
                "graph_probability": graph_probability, "chemical_probability": chemical_probability,
                "probability": probability, "model_prediction": raw, "prediction": prediction}

    def predict(self, query, unlabeled=None):
        return self.predict_components(query, unlabeled)["prediction"]
