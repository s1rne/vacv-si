"""Reproduce the frozen train-only Ridge-log artifact locally; never upload."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.v2_analog import fingerprint, tanimoto
from experiments.v2_context_analog import contexts, same_context
from experiments.v4_pair_delta import fit_predict
from vacv.dataset import load, submission

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments/results/v4-pair-delta-submission/submission-pair-delta-v4.csv"
FROZEN = ROOT / "experiments/results/v4-pair-delta/frozen-winner.json"
RTOL = ATOL = 1e-12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_submission(frame, template, expected_rows=597):
    """Check the full artifact contract, including exact sample index order."""
    columns = ["index", "SI VACV"]
    if list(frame.columns) != columns or list(template.columns) != columns:
        raise ValueError("Submission and sample must have columns index, SI VACV")
    if len(frame) != expected_rows or len(template) != expected_rows:
        raise ValueError(f"Expected {expected_rows} submission and sample rows")
    submission(frame["index"].to_numpy(), frame["SI VACV"].to_numpy())
    if not np.array_equal(frame["index"].to_numpy(), template["index"].to_numpy()):
        raise ValueError("Submission indices differ from sample index order")


def reproduce(data_folder, out):
    data_folder, out = Path(data_folder), Path(out)
    verification = out.with_suffix(".verification.json")
    for path in (out, verification):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if json.loads(FROZEN.read_text())["winner"] != "ridge-log":
        raise ValueError("Frozen winner must be ridge-log")
    # The existing loader explicitly selects allowed fields and only train labels.
    data = load(data_folder, require_v2=True)
    if len(data.train) != 320 or len(data.test) != 597:
        raise ValueError("Expected official v2: 320 training and 597 test rows")
    if not np.isfinite(data.y).all() or (data.y < 0).any():
        raise ValueError("Training targets must be finite and nonnegative")
    template = pd.read_csv(data_folder / "sample_submit.csv")
    validate_submission(submission(data.test["index"], np.zeros(len(data.test))), template)
    frames = pd.concat([data.train, data.test], ignore_index=True)
    x, ctx = fingerprint(frames), contexts(frames)
    tr = np.arange(len(data.train))
    query = np.arange(len(data.train), len(frames))
    with threadpool_limits(limits=2):
        predictions, fired, pairs = fit_predict(
            "ridge-log", x, data.y, tanimoto(x, x), same_context(ctx, ctx), tr, query)
    result = submission(data.test["index"].to_numpy(), predictions)
    validate_submission(result, template)
    reference = pd.read_csv(REFERENCE)
    validate_submission(reference, template)
    np.testing.assert_allclose(result["SI VACV"], reference["SI VACV"], rtol=RTOL, atol=ATOL)
    # Exclusive creation also protects against an output created during inference.
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)
    reread = pd.read_csv(out)
    validate_submission(reread, template)
    np.testing.assert_allclose(reread["SI VACV"], predictions, rtol=RTOL, atol=ATOL)
    source_paths = [Path(__file__), ROOT / "scripts/submit_pair_delta_v4.py",
                    ROOT / "experiments/v4_pair_delta.py", ROOT / "experiments/v2_analog.py",
                    ROOT / "experiments/v2_context_analog.py", ROOT / "vacv/dataset.py",
                    ROOT / "vacv/pipeline.py", ROOT / "vacv/curate.py", ROOT / "vacv/splits.py", FROZEN]
    report = {
        "profile": "ridge-log", "training_rows": len(tr), "test_rows": len(query),
        "directed_training_pairs": pairs, "pair_expert_rows": int(fired.sum()),
        "external_data": False, "test_labels_used": False, "forbidden_features_used": False,
        "uploaded": False,
        "comparison": {"reference": str(REFERENCE.relative_to(ROOT)),
                       "numeric_equal_within_tolerance": True, "rtol": RTOL, "atol": ATOL,
                       "max_absolute_difference": float(np.max(np.abs(reread["SI VACV"] - reference["SI VACV"]))),
                       "byte_equal": sha256(out) == sha256(REFERENCE),
                       "output_sha256": sha256(out), "reference_sha256": sha256(REFERENCE)},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        "input_sha256": {name: sha256(data_folder / name) for name in ("train_v2.csv", "test.csv", "sample_submit.csv")},
        "dependencies": {"python": platform.python_version(), **{name: version(name) for name in
                         ("numpy", "pandas", "scikit-learn", "scipy", "rdkit", "threadpoolctl")}},
        "historical_public_score": {"value": 1710149.26680, "observed_date": "2026-11-07",
                                    "provenance": "experiments/results/v4-pair-delta-submission/result.md",
                                    "provenance_sha256": sha256(REFERENCE.parent / "result.md"),
                                    "newly_measured": False},
    }
    with verification.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw_v2"))
    parser.add_argument("--out", type=Path, default=Path("outputs/submission-pair-delta-v4.csv"))
    args = parser.parse_args()
    print(json.dumps(reproduce(args.data, args.out), indent=2))


if __name__ == "__main__":
    main()
