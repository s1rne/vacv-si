"""Predeclared pair-delta representations; unchanged v4 pair and anchor selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v4_pair_delta import fit_predict, training_pairs
from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from experiments.v2_context_analog import contexts, same_context
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = ("morgan-r3-bit", "morgan-r2-logcount", "morgan-r2-r3-bit")
BASELINE = "v4-ridge-log-r2-bit"
OUT = Path("experiments/results/v5-pair-features")


def representations(frame):
    """Structure-only deterministic features, no label/context arguments."""
    gen2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    gen3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    counts, radius3 = [], []
    for smiles in frame.SMILES:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        counts.append(np.asarray(gen2.GetCountFingerprintAsNumPy(mol), float)
                      if mol is not None else np.zeros(2048))
        radius3.append(np.asarray(gen3.GetFingerprintAsNumPy(mol), float)
                       if mol is not None else np.zeros(2048))
    c, r3 = np.asarray(counts), np.asarray(radius3)
    return {PROFILES[0]: r3, PROFILES[1]: np.log1p(c),
            PROFILES[2]: np.concatenate(((c > 0).astype(float), r3), axis=1) / np.sqrt(2)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    y, original = data.y, fingerprint(data.train)
    xs = {BASELINE: original, **representations(data.train)}
    sim = tanimoto(original, original)
    ctx = contexts(data.train)
    matching = same_context(ctx, ctx)
    metrics, audits, winner = [], [], None
    for split in (41, 42, 43, 44, 45, "scaffold"):
        names = (*PROFILES, BASELINE) if split == 41 else (winner, BASELINE)
        if split == "scaffold":
            groups = scaffold_folds(scaffolds_of(data.train), 5)
            iterator = (fold_indices(groups, f) for f in range(5))
        else:
            iterator = KFold(5, shuffle=True, random_state=split).split(y)
        predictions = {n: np.full(len(y), np.nan) for n in names}
        fired = {n: np.zeros(len(y), bool) for n in names}
        folds = np.full(len(y), -1)
        for fold, (tr, va) in enumerate(iterator):
            folds[va] = fold
            pairs = training_pairs(sim, matching, tr)
            assert not np.isin(pairs, va).any()
            audits.append({"split": split, "fold": fold, "train_rows": tr.tolist(),
                           "validation_rows": va.tolist(), "directed_pairs": len(pairs),
                           "pair_rows": np.unique(pairs).tolist()})
            for name in names:
                p, gate, _ = fit_predict("ridge-log", xs[name], y, sim, matching, tr, va)
                predictions[name][va], fired[name][va] = p, gate
            print(f"split={split} fold={fold} done", flush=True)
        for name, p in predictions.items():
            assert np.isfinite(p).all()
            metrics.append({"split": split, "model": name, **diagnostics(y, p),
                            "pair_expert_rows": int(fired[name].sum())})
        pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": folds,
                      **predictions, **{n + "-fired": fired[n] for n in names}}).to_csv(OUT / f"oof-{split}.csv", index=False)
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if split == 41:
            winner = min((m for m in metrics if m["model"] in PROFILES), key=lambda m: m["mse"])["model"]
            (OUT / "frozen-winner.json").write_text(json.dumps({"winner": winner, "discovery_seed": 41}) + "\n")
        print(json.dumps(metrics[-len(names):]), flush=True)
    confirm = [m for m in metrics if m["split"] in (42, 43, 44, 45) and m["model"] == winner]
    base = [m for m in metrics if m["split"] in (42, 43, 44, 45) and m["model"] == BASELINE]
    avg, avgbase = float(np.mean([m["mse"] for m in confirm])), float(np.mean([m["mse"] for m in base]))
    manifest = {"winner": winner, "profiles": PROFILES, "discovery": 41, "confirmations": [42, 43, 44, 45],
                "confirmation_mean_mse": avg, "paired_v4_mean_mse": avgbase,
                "improves_all_confirmations": all(m["mse"] < b["mse"] for m, b in zip(confirm, base)),
                "each_confirmation_below_3m": all(m["mse"] <= 3e6 for m in confirm),
                "rows": len(y), "max_target": float(y.max()), "rdkit_version": rdBase.rdkitVersion,
                "similarity_representation": "unchanged original Morgan radius2 binary2048",
                "external_data": False, "forbidden_features": False, "submission_created": False,
                "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__), Path("experiments/v4_pair_delta.py"),
                            Path("experiments/v2_analog.py"), Path("experiments/v2_context_analog.py"),
                            Path("vacv/pipeline.py"), Path("vacv/features.py"), Path("data/raw_v2/train_v2.csv")]}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "fold-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    lines = ["# Pair-delta molecular representation experiment", "",
             f"Frozen winner on seed41: {winner}. Paired control is v4 Ridge-log, the model family associated with the user's reported ~1.710M public score; local CV is not that public score.", "",
             "| Split | Representation | Full raw MSE |", "|---|---|---:|"]
    lines += [f"| {m['split']} | {m['model']} | {m['mse']:.3f} |" for m in metrics]
    lines += ["", f"Confirmation mean: {avg:.3f}; paired v4 mean: {avgbase:.3f}; relative change: {(avg/avgbase-1)*100:.3f}%.",
              f"Improves every confirmation: {manifest['improves_all_confirmations']}. Each confirmation <=3M: {manifest['each_confirmation_below_3m']}.", "",
              "Only the delta vector changes: radius3 2048-bit, radius2 log1p hashed counts2048, or concatenated radius2/radius3 binary vectors with each block scaled by 1/sqrt(2). All use existing v4 Ridge alpha10, no intercept, log1p target deltas, antisymmetry, same training pairs, top3 anchors, original radius2 similarity thresholds and fallback. No hyperparameter search. Every label remains in full raw MSE, without target cap/removal. All learned pair labels restricted to outer train. No external data, forbidden fields, candidate CSV or uploads. Scaffold is a stress test.", ""]
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
