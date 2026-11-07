"""Target-independent external counts and similarity, without recovered SI."""

import numpy as np
import pandas as pd

from .literature import PRESENCE_FEATURES, presence


def attach_clean(frame):
    """Preserve input order; never read the label-fitted lookup artifact."""
    allowed = [c for c in ("index", "SMILES", "Title", "strain", "Culture") if c in frame]
    result = frame[allowed].merge(presence(), on="SMILES", how="left",
                                  sort=False, validate="many_to_one")
    for column in PRESENCE_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["matched"] = 0.0
    result["literature_si"] = np.nan
    result["literature_quality"] = None
    return result
