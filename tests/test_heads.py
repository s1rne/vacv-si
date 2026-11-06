import numpy as np
import pandas as pd

from vacv.endpoints import CONTEXT_COLUMNS, load_corpus
from vacv.heads import HEAD_FEATURES, EndpointHeads, competition_matrix, corpus_matrix
from vacv.multitask import MultitaskModel
from vacv.pipeline import SENTINEL

FRAME = pd.DataFrame({
    "SMILES": ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1CCC[C@H]1c1cccnc1"],
    "Culture": ["Vero", "HeLa cells", "cells HEL", "Vero"],
    "strain": ["Vaccinia virus WR", "Vaccinia virus", "Vaccinia virus", "Vaccinia virus"],
    "Title": ["1", "2a", "3", "4"],
})


def test_matrices_share_the_structure_and_context_block():
    corpus = load_corpus().head(20)
    assert corpus_matrix(corpus).shape[1] == competition_matrix(FRAME).shape[1]
    assert corpus_matrix(corpus).shape[1] > len(CONTEXT_COLUMNS)


def test_heads_predict_both_endpoints_and_the_ratio():
    heads = EndpointHeads().fit(load_corpus().head(200))
    predicted = heads.predict(FRAME)
    assert list(predicted.columns) == ["log_ic50", "log_cc50", "log_si", "p_inactive"]
    assert set(HEAD_FEATURES) == set(predicted.columns)
    assert np.allclose(predicted["log_si"], predicted["log_cc50"] - predicted["log_ic50"])
    assert predicted["p_inactive"].between(0, 1).all()


def test_dropping_censored_rows_leaves_fewer_training_labels():
    corpus = load_corpus().head(300)
    assert EndpointHeads(censored=False).fit(corpus).scale_["ic50"] != \
        EndpointHeads(censored=True).fit(corpus).scale_["ic50"]


def test_multitask_prediction_stays_inside_the_target_range():
    heads = EndpointHeads().fit(load_corpus().head(200))
    y = np.array([SENTINEL, 5.0, 12.0, SENTINEL])
    model = MultitaskModel(heads).fit(FRAME, y)
    prediction = model.predict(FRAME)
    assert ((prediction > 0) & (prediction <= SENTINEL)).all()
    assert model.target_prior_ == 0.5
