"""Freeze a provisional numeric-data ablation, not a biological source validation."""
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from vacv.curate import standardize
from vacv.dataset import load

COMMIT = "066ad8bf2f48afc9d469a5925cde2ed9676541d0"
OUT = Path("experiments/results/v9-similar-snapshot")


def record_pair(value):
    if value.startswith("CHEMBL_ACT:"):
        ids = re.findall(r"CHEMBL_ACT:(\d+)", value)
    elif "activity " in value:
        ids = re.findall(r"activity (\d+)", value)
    elif re.fullmatch(r"CHEMBL\d+:\d+:\d+", value):
        ids = value.split(":")[-2:]
    elif re.fullmatch(r"\d+/\d+", value):
        ids = value.split("/")
    else:
        return None
    return "/".join(ids[:2]) if len(ids) == 2 else None


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    raw = subprocess.check_output(["git", "show", f"{COMMIT}:data/external/similar_all.csv"])
    frame = pd.read_csv(io.BytesIO(raw))
    data = load(Path("data/raw_v2"), require_v2=True)
    cache = {}

    def key(s):
        if s not in cache:
            value = standardize(s)
            cache[s] = value.inchikey[:14] if value.inchikey else None
        return cache[s]

    blocked = set(pd.concat([data.train.SMILES, data.test.SMILES]).map(key).dropna())
    frame["key"] = frame.smiles.map(key)
    frame["record_pair"] = frame.source_id.map(record_pair)
    frame["source_row"] = np.arange(len(frame))
    reasons = []
    for row in frame.itertuples():
        rejected = []
        if pd.isna(row.key):
            rejected.append("invalid_identity")
        if row.key in blocked:
            rejected.append("competition_identity")
        if row.ic50_relation != "=" or row.cc50_relation != "=":
            rejected.append("nonexact_measurement")
        if not np.isfinite([row.ic50_um, row.cc50_um]).all() or min(row.ic50_um, row.cc50_um) <= 0:
            rejected.append("nonpositive_or_nonfinite")
        if pd.isna(row.record_pair) or not row.source.startswith("ChEMBL"):
            rejected.append("unparsed_record_pair")
        reasons.append(rejected)
    compare = ["key", "ic50_um", "cc50_um", "ic50_relation", "cc50_relation", "doi"]
    for pair, indices in frame.groupby("record_pair").groups.items():
        if any(frame.loc[indices, c].nunique(dropna=False) > 1 for c in compare):
            for i in indices:
                reasons[i].append("conflicting_record_pair")
    eligible = frame.loc[[not r for r in reasons]].drop_duplicates("record_pair").copy()
    eligible["target"] = eligible.cc50_um / eligible.ic50_um
    counts = eligible.groupby("key").key.transform("size")
    eligible["weight"] = .25 / counts
    eligible = eligible.rename(columns={"smiles": "SMILES"})
    assert len(eligible) and np.isfinite(eligible.target).all()
    assert not set(eligible.key) & blocked
    OUT.mkdir(parents=True)
    (OUT / "source.csv").write_bytes(raw)
    eligible[["source_row", "SMILES", "target", "weight", "key", "record_pair", "doi", "source"]].to_csv(
        OUT / "provisional-exact.csv", index=False)
    pd.DataFrame({"source_row": frame.source_row, "reasons": ["|".join(r) for r in reasons]}).to_csv(
        OUT / "exclusions.csv", index=False)
    info = {"commit": COMMIT, "source_rows": len(frame), "provisional_rows": len(eligible),
            "provisional_identities": eligible.key.nunique(), "competition_overlap_rows": int(frame.key.isin(blocked).sum()),
            "unflagged_overlap_rows": int((frame.key.isin(blocked) & ~frame.is_competition_compound).sum()),
            "competition_overlap_after_filter": 0, "weight_per_external_identity": .25,
            "interpretation": "Provisional numerical ablation ONLY; source endpoint/cell/exposure compatibility unverified",
            "limitations": ["Input mixes endpoint types; no independent source-record audit",
                            "Input lacks paired cell/exposure metadata; known different-cell pairs may remain",
                            "Not eligible for promotion/deployment without separate provenance review"],
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "snapshot_sha256": hashlib.sha256((OUT / "provisional-exact.csv").read_bytes()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
