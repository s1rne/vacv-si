"""Audited disjoint external exact-SI pairs augment v4 delta learning only."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v3_intervals import read_snapshot
from experiments.v4_pair_delta import training_pairs, antisymmetric_predict, fit_predict as baseline
from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from experiments.v2_context_analog import contexts, same_context, predict as analog
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

SNAPSHOT = Path("experiments/results/v3-source-pairs/eligible.csv")
SHA = "9ead9fc63139b88352584a39673f3458beaf1abb8c3ddd776c0de7d1729a2e23"
PROFILES = {"external10": (False, .1), "cross10": (True, .1), "cross50": (True, .5)}
BASE = "v4-ridge-log"
OUT = Path("experiments/results/v6-pair-external")


def load_external(train, test, path=SNAPSHOT, expected_sha=SHA):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("External immutable snapshot SHA mismatch")
    clean, excluded = read_snapshot(path, train, test)
    # Read source context only after hash validation and independent overlap filter.
    context = pd.read_csv(path, usecols=["SMILES", "strain", "cell"])
    clean = clean.loc[clean.kind.eq("exact")].merge(context, on="SMILES", validate="one_to_one")
    clean = clean.rename(columns={"cell": "Culture"}).reset_index(drop=True)
    return clean, excluded


def source_context(frame):
    """Conservative named strain/cell aliases, independent of outcomes."""
    result = []
    cells = {"vero": "vero", "hff": "hff", "human foreskin fibroblast": "hff",
             "bsc-1": "bsc-1", "bsc1": "bsc-1"}
    for strain, cell in frame[["strain", "Culture"]].fillna("").itertuples(index=False, name=None):
        text = str(strain).strip().casefold()
        named = [name for name, pattern in (("copenhagen", r"\bcopenhagen\b"),
                  ("wr", r"\bwestern reserve\b|\bwr\b"), ("lister", r"\blister\b"))
                 if re.search(pattern, text)]
        mixed = re.search(r"cowpox|ectromelia|grishak|tiantan|ihd-j|lederle", text)
        strain_id = named[0] if len(named) == 1 and not mixed else ""
        result.append((strain_id, cells.get(str(cell).strip().casefold(), "")))
    return np.asarray(result)


def fit_predict(profile, x, y, sim, original_match, source_match, tr, va, n_comp):
    ext = np.arange(n_comp, len(y))
    cp = training_pairs(sim, original_match, tr)
    ep = training_pairs(sim, source_match, ext)
    cross, weight = PROFILES[profile]
    xp = np.empty((0, 2), int)
    if cross:
        crossmatch = source_match.copy()
        crossmatch[:n_comp, :n_comp] = False
        crossmatch[n_comp:, n_comp:] = False
        xp = training_pairs(sim, crossmatch, np.r_[tr, ext])
    pairs = np.vstack((cp, ep, xp))
    assert not np.isin(pairs, va).any()
    weights = np.r_[np.ones(len(cp)), np.full(len(ep) + len(xp), weight)]
    # Only train/external targets are indexed; validation labels never enter fitting.
    targets = np.log1p(y[np.r_[tr, ext]])
    local = {int(row): i for i, row in enumerate(np.r_[tr, ext])}
    a = np.array([local[int(i)] for i in pairs[:, 0]], dtype=int)
    b = np.array([local[int(i)] for i in pairs[:, 1]], dtype=int)
    delta = targets[a] - targets[b]
    high = ext[y[ext] >= 1000]
    highpairs = np.isin(pairs, high).any(axis=1)
    audit = {"competition_pairs": len(cp), "external_pairs": len(ep), "cross_pairs": len(xp),
             "pair_rows": np.unique(pairs).tolist(), "external_used_rows": np.unique(pairs[pairs >= n_comp]).tolist(),
             "high_external_directed_pairs": int(highpairs.sum()),
             "high_external_weighted_delta_squared": float(np.sum(weights[highpairs] * delta[highpairs] ** 2)),
             "all_weighted_delta_squared": float(np.sum(weights * delta ** 2))}
    query_sim, match = sim[np.ix_(va, tr)], original_match[np.ix_(va, tr)]
    pred = analog(query_sim, np.zeros_like(match), y[tr], "context-log70")[0]
    fired = np.zeros(len(va), bool)
    if not len(pairs):
        return pred, fired, audit
    model = Ridge(alpha=10., fit_intercept=False).fit(x[pairs[:, 0]] - x[pairs[:, 1]], delta, sample_weight=weights)
    train_targets = np.log1p(y[tr])
    for k, row in enumerate(va):
        anchors = np.flatnonzero(match[k] & (query_sim[k] >= .7))
        anchors = anchors[np.argsort(-query_sim[k, anchors], kind="stable")[:3]]
        if not len(anchors):
            continue
        changes = antisymmetric_predict(model, x[row][None, :] - x[tr[anchors]])
        values = np.expm1(train_targets[anchors] + changes)
        w = query_sim[k, anchors] ** 8
        pred[k] = max(0., float(w @ values / w.sum()))
        fired[k] = True
    assert np.isfinite(pred).all()
    return pred, fired, audit


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    data = load(Path("data/raw_v2"), require_v2=True)
    external, excluded = load_external(data.train, data.test)
    assert len(external) == 47
    external.to_csv(OUT / "external-exact.csv", index=False)
    n = len(data.y)
    frame = pd.concat([data.train, external], ignore_index=True)
    x, y = fingerprint(frame), np.r_[data.y, external.lower_si.to_numpy(float)]
    sim, ctx = tanimoto(x, x), contexts(frame)
    match = same_context(ctx, ctx)
    sc = source_context(frame)
    source_match = same_context(sc, sc)
    metrics, audits, winner = [], [], None
    for split in (51, 52, 53, 54, 55, "scaffold"):
        names = (*PROFILES, BASE) if split == 51 else (winner, BASE)
        if split == "scaffold":
            groups = scaffold_folds(scaffolds_of(data.train), 5)
            iterator = (fold_indices(groups, f) for f in range(5))
        else:
            iterator = KFold(5, shuffle=True, random_state=split).split(data.y)
        outputs = {name: np.full(n, np.nan) for name in names}
        folds = np.full(n, -1)
        for fold, (tr, va) in enumerate(iterator):
            folds[va] = fold
            for name in names:
                if name == BASE:
                    pred, _, _ = baseline("ridge-log", x, y, sim, match, tr, va)
                else:
                    pred, _, audit = fit_predict(name, x, y, sim, match, source_match, tr, va, n)
                    audits.append({"split": split, "fold": fold, "profile": name,
                                   "train_rows": tr.tolist(), "validation_rows": va.tolist(), **audit})
                outputs[name][va] = pred
            print(f"split={split} fold={fold}", flush=True)
        for name, p in outputs.items():
            metrics.append({"split": split, "model": name, **diagnostics(data.y, p)})
        pd.DataFrame({"row": np.arange(n), "target": data.y, "fold": folds, **outputs}).to_csv(OUT / f"oof-{split}.csv", index=False)
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if split == 51:
            winner = min((m for m in metrics if m["model"] in PROFILES), key=lambda m: m["mse"])["model"]
            (OUT / "frozen-winner.json").write_text(json.dumps({"winner": winner, "discovery": 51}) + "\n")
        print(json.dumps(metrics[-len(names):]), flush=True)
    confirmations = [m for m in metrics if m["split"] in (52, 53, 54, 55) and m["model"] == winner]
    bases = [m for m in metrics if m["split"] in (52, 53, 54, 55) and m["model"] == BASE]
    manifest = {"winner": winner, "profiles": PROFILES, "discovery": 51, "confirmation": [52, 53, 54, 55],
                "confirmation_mean_mse": float(np.mean([m["mse"] for m in confirmations])),
                "paired_v4_mean_mse": float(np.mean([m["mse"] for m in bases])),
                "improves_all": all(a["mse"] < b["mse"] for a, b in zip(confirmations, bases)),
                "all_confirmations_below3m": all(m["mse"] <= 3e6 for m in confirmations),
                "external_exact_rows": len(external), "external_known_context_rows": int((sc[n:] != "").all(1).sum()),
                "external_max_si": float(external.lower_si.max()), "external_si_over1000": int((external.lower_si >= 1000).sum()),
                "excluded_snapshot_rows": len(excluded), "bounds_used": False, "query_external_anchors": False,
                "competition_rows": n, "max_competition_si": float(data.y.max()), "no_candidate_or_upload": True,
                "snapshot_sha256": SHA,
                "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__), Path("experiments/v3_intervals.py"), Path("experiments/v4_pair_delta.py"),
                            Path("experiments/v2_analog.py"), Path("experiments/v2_context_analog.py"), Path("data/raw_v2/train_v2.csv")]}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "fold-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    lines = ["# External exact-SI pair transfer", "", f"Frozen seed51 winner: {winner}.", "",
             "| Split | Profile | Full raw MSE |", "|---|---|---:|"]
    lines += [f"| {m['split']} | {m['model']} | {m['mse']:.3f} |" for m in metrics]
    lines += ["", json.dumps(manifest, indent=2), "",
              "Immutable snapshot hash checked before reading context; read_snapshot independently blocks all competition train/test key14 and exact SMILES plus duplicate external identities. Only47 exact rows retained, no bounds converted to labels, no SI cap. Competition pair selection unchanged. External-only pairs and optional cross pairs require known named strain/cell and original similarity>=.5 top8 in either direction. Conservative aliases: Copenhagen, WR/Western Reserve, Lister; Vero, HFF/Human foreskin fibroblast, BSC-1/BSC1; unknown remains unavailable. Competition original context gate and query anchors remain unchanged, outer-train competition only. No external lookup or direct prediction substitution. Training pair weights:1 competition, .1 or .5 external-containing; alpha10 unchanged. Pair audits record support and high-external-target weighted squared-delta influence. Full320 raw MSE, no target removal/clipping. Discovery51 freezes before52–55; scaffold stress. No submissions generated or uploaded.", ""]
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
