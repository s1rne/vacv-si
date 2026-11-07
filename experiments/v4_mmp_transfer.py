"""Single-cut MMP transformation transfer; fixed profiles and honest outer folds."""
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdMMPA
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v2_context_analog import contexts, same_context, predict as analog
from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = ("mmp-log", "mmp-raw")
OUT = Path("experiments/results/v4-mmp-transfer")


def fragments(smiles):
    """Canonical mapped one-attachment fragments, with a dominant connected core."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ()
    size = mol.GetNumHeavyAtoms()
    result = set()
    # Explicit bond selection avoids FragmentMol's default 20-bond size exclusion.
    bonds = [b.GetIdx() for b in mol.GetBonds()
             if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE
             and b.GetBeginAtom().GetAtomicNum() > 1
             and b.GetEndAtom().GetAtomicNum() > 1]
    if not bonds:
        return ()
    for _, sides in rdMMPA.FragmentMol(mol, bonds, 1, 1, False):
        parts = Chem.GetMolFrags(Chem.MolFromSmiles(sides), asMols=True)
        if len(parts) != 2:
            continue
        for core, variable in (parts, parts[::-1]):
            if core.GetNumHeavyAtoms() * 2 < size or variable.GetNumHeavyAtoms() > 8:
                continue
            if any(sum(a.GetAtomicNum() == 0 for a in p.GetAtoms()) != 1
                   for p in (core, variable)):
                continue
            for p in (core, variable):
                for a in p.GetAtoms():
                    a.SetAtomMapNum(1 if a.GetAtomicNum() == 0 else 0)
            result.add((Chem.MolToSmiles(core), Chem.MolToSmiles(variable)))
    return tuple(sorted(result))


class MMPExpert:
    """Fit receives only outer-training structures, targets and audit row IDs."""
    def fit(self, frags, ctx, y, row_ids, profile):
        self.profile = profile
        self.targets = np.log1p(y) if profile == "mmp-log" else np.asarray(y) / 1000
        self.row_ids = np.asarray(row_ids)
        self.cores = defaultdict(list)
        self.transforms = defaultdict(dict)
        for i, (entries, context) in enumerate(zip(frags, ctx)):
            if not all(context):
                continue
            for core, variable in entries:
                self.cores[(core, tuple(context))].append((i, variable))
        for (_, context), anchors in self.cores.items():
            for (i, a), (j, b) in combinations(anchors, 2):
                if i == j or a == b:
                    continue
                pair = tuple(sorted((int(self.row_ids[i]), int(self.row_ids[j]))))
                delta = float(self.targets[j] - self.targets[i])
                self.transforms[(a, b, context)][pair] = delta
                self.transforms[(b, a, context)][pair] = -delta
        return self

    def predict(self, frags, ctx, fallback):
        prediction = np.asarray(fallback).copy()
        fired = np.zeros(len(frags), dtype=bool)
        supports = np.zeros(len(frags), dtype=int)
        for qi, (entries, context) in enumerate(zip(frags, ctx)):
            proposals = {}
            for core, variable in entries:
                for anchor, avar in self.cores.get((core, tuple(context)), ()):
                    changes = self.transforms.get((avar, variable, tuple(context)), {})
                    if len(changes) < 2:
                        continue
                    n = len(changes)
                    shrink = n / (n + 5.)
                    value = self.targets[anchor] + shrink * np.mean(list(changes.values()))
                    value = max(0., np.expm1(value) if self.profile == "mmp-log" else value * 1000.)
                    proposals[(anchor, avar, variable)] = (value, shrink, n)
            if proposals:
                values, weights, counts = np.asarray(list(proposals.values())).T
                prediction[qi] = np.average(values, weights=weights)
                fired[qi] = True
                supports[qi] = int(max(counts))
        return prediction, fired, supports


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    y, ctx = data.y, contexts(data.train)
    frags = [fragments(s) for s in data.train.SMILES]
    x = fingerprint(data.train)
    sim, match = tanimoto(x, x), same_context(ctx, ctx)
    metrics, audits, winner = [], [], None
    for split in (36, 37, 38, "scaffold"):
        names = PROFILES if split == 36 else (winner,)
        folds = (KFold(5, shuffle=True, random_state=split).split(y)
                 if split != "scaffold" else
                 (fold_indices(scaffold_folds(scaffolds_of(data.train), 5), f) for f in range(5)))
        predictions = {n: np.full(len(y), np.nan) for n in names}
        fired = {n: np.zeros(len(y), bool) for n in names}
        counts = {n: np.zeros(len(y), int) for n in names}
        control, fold_ids = np.full(len(y), np.nan), np.full(len(y), -1)
        for fold, (tr, va) in enumerate(folds):
            base = analog(sim[np.ix_(va, tr)], match[np.ix_(va, tr)], y[tr], "context-raw70")[0]
            control[va], fold_ids[va] = base, fold
            for name in names:
                model = MMPExpert().fit([frags[i] for i in tr], ctx[tr], y[tr], tr, name)
                label_rows = {i for changes in model.transforms.values() for pair in changes for i in pair}
                assert not label_rows.intersection(va)
                p, f, c = model.predict([frags[i] for i in va], ctx[va], base)
                predictions[name][va], fired[name][va], counts[name][va] = p, f, c
                audits.append({"split": split, "fold": fold, "profile": name,
                               "train_rows": tr.tolist(), "validation_rows": va.tolist(),
                               "transformation_label_rows": sorted(label_rows),
                               "supported_transformations": sum(len(v) >= 2 for v in model.transforms.values())})
        control_mse = float(np.mean((y - control) ** 2))
        for name, p in predictions.items():
            assert np.isfinite(p).all()
            metrics.append({"split": split, "model": name, **diagnostics(y, p),
                            "control_mse": control_mse, "fired": int(fired[name].sum()),
                            "max_target_fired": bool(fired[name][np.argmax(y)])})
        pd.DataFrame({"row": np.arange(len(y)), "target": y, "fold": fold_ids,
                      "context-raw70": control, **predictions,
                      **{n + "-fired": fired[n] for n in names},
                      **{n + "-supports": counts[n] for n in names}}).to_csv(OUT / f"oof-{split}.csv", index=False)
        if split == 36:
            winner = min(metrics, key=lambda m: m["mse"])["model"]
        print(json.dumps(metrics[-len(names):]), flush=True)
    confirmation = [m for m in metrics if m["split"] in (37, 38)]
    research_gate = (np.mean([m["mse"] for m in confirmation]) < 4e6
                     and all(m["mse"] < m["control_mse"] for m in confirmation))
    kaggle_gate = bool(research_gate and all(m["mse"] <= 3e6 for m in confirmation))
    manifest = {"winner_frozen_after_seed36": winner, "confirmation_seeds": [37, 38],
                "research_gate_passed": bool(research_gate), "kaggle_gate_passed": kaggle_gate,
                "training_rows": len(y), "max_target": float(max(y)), "rdkit_version": rdBase.rdkitVersion,
                "fragmented_rows": sum(bool(f) for f in frags), "profiles": PROFILES,
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [Path(__file__), Path("experiments/v2_context_analog.py"),
                                   Path("data/raw_v2/train_v2.csv")]},
                "no_external_data": True, "no_submission_created": True}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "fold-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    lines = ["# Single-cut MMP transfer", "", "## Measured results", "",
             "| Split | Model | Raw MSE | Paired context control | Covered rows |",
             "|---|---|---:|---:|---:|"]
    lines += [f"| {m['split']} | {m['model']} | {m['mse']:.3f} | {m['control_mse']:.3f} | {m['fired']}/320 |" for m in metrics]
    lines += ["", f"Frozen winner: {winner}. Research gate: {bool(research_gate)}. Kaggle gate: {kaggle_gate}.",
              "No candidate CSV or Kaggle upload. All targets retained without clipping or removal.", "",
              "## Method and limitations", "",
              "Actual RDKit single non-ring bond cuts, connected core >= half original heavy atoms, variable <=8 heavy atoms; atom-mapped attachment retained. Canonical pairs require identical core and known strain/culture. Directed variable substitutions require two unique unordered training pairs within that same context. Mean transformed-target difference is shrunk n/(n+5), added to a training anchor, then inversed; support-weighted arithmetic mean of proposals. Missing transfer falls back to paired context-raw70. No query targets enter the fitting API. All 320 rows have outer-fold predictions; discovery only selects between two fixed profiles before confirmations. Scaffold is a stress test, not a tuning split.",
              "", "This is a narrow custom MMP expert, not a reproduction of mmpdb's full multi-cut workflow. Sparse 320-row SI data and assay context restrictions may leave too few repeated transformations. Literature on larger activity datasets does not establish raw-SI score improvements here. Supports can share molecules and are not independent experimental replicates. No external labels or forbidden columns used.", "",
              "## Primary sources", "",
              "- mmpdb: https://pubs.acs.org/doi/10.1021/acs.jcim.8b00173 ; official implementation https://github.com/rdkit/mmpdb",
              "- MoleculeACE activity-cliff evaluation: https://pmc.ncbi.nlm.nih.gov/articles/PMC9749029/",
              "- Activity-cliff prediction comparison: https://pmc.ncbi.nlm.nih.gov/articles/PMC10107580/", ""]
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
