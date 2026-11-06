import numpy as np
import pandas as pd

from vacv.endpoints import (CELL_COLUMNS, STRAIN_COLUMNS, bounds, competition_context,
                            context_frame, load_corpus)
from vacv.multitask import shift_prior


def test_bounds_encode_relations():
    lower, upper = bounds(np.array([1.0, 2.0, 3.0, 4.0]), np.array(["=", ">", ">=", "<"], dtype=object))
    assert list(lower) == [1.0, 2.0, 3.0, 0.0]
    assert upper[0] == 1.0 and np.isinf(upper[1]) and np.isinf(upper[2]) and upper[3] == 4.0


def test_cell_line_vocabulary_separates_hela_from_hel():
    frame = pd.DataFrame({"cell": ["HELA cells", "cells HEL", "MT-4|Vero", None], "strain": ["WR"] * 4})
    encoded = context_frame(frame, "cell", "strain")
    assert encoded.loc[0, "cell_hela"] == 1 and encoded.loc[0, "cell_hel"] == 0
    assert encoded.loc[1, "cell_hel"] == 1 and encoded.loc[1, "cell_hela"] == 0
    assert encoded.loc[2, ["cell_mt4", "cell_vero"]].tolist() == [1.0, 1.0]
    assert encoded.loc[3, CELL_COLUMNS].sum() == 0
    assert encoded.loc[0, "strain_wr"] == 1


def test_competition_context_reads_culture_out_of_strain_text():
    frame = pd.DataFrame({
        "Culture": ["Vero", " lung fibroblast cells"],
        "strain": ["Vaccinia virus (strain Copenhagen). culture Vero", "Vaccinia virus Tiantan strain"],
    })
    encoded = competition_context(frame)
    assert list(encoded.columns) == CELL_COLUMNS + STRAIN_COLUMNS
    assert encoded.loc[0, "cell_vero"] == 1 and encoded.loc[0, "strain_copenhagen"] == 1
    assert encoded.loc[1, "cell_hel"] == 1 and encoded.loc[1, "strain_tiantan"] == 1


def test_corpus_labels_follow_the_censoring_flags():
    corpus = load_corpus()
    assert (corpus["inactive"] == corpus["ic50_censored"].astype(int)).all()
    censored = corpus["ic50_censored"].to_numpy(bool)
    assert np.isinf(corpus.loc[censored, "ic50_upper"]).all()
    assert np.isfinite(corpus.loc[~censored, "ic50_upper"]).all()
    assert np.allclose(corpus["log_si"], corpus["log_cc50"] - corpus["log_ic50"])


def test_prior_shift_is_identity_at_equal_prevalence():
    probability = np.array([0.1, 0.5, 0.9])
    assert np.allclose(shift_prior(probability, 0.3, 0.3), probability)
    lowered = shift_prior(probability, 0.6, 0.25)
    assert (lowered < probability).all()
