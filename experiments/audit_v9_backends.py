"""Independently audit saved V9 artifacts; never import or run fitting code.

Run from the repository root with ``python -m experiments.audit_v9_backends``.
A completed manifest is required. Only independent-audit.json is written;
protocol, sources, inputs and prediction artifacts are read-only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


EXPECTED_PROTOCOL = "6117a8480f64983017b5428b85e361e71a4ca62cb5a7caf12940a440e902a2d9"
MODELS = ["v4", "et-raw", "et-log", "cat-raw", "cat-log", "kernel-raw",
          "kernel-log", "et-log-external"]
SCHEMES = ["identity", "identity71", "identity72", "scaffold"]
NAMES = MODELS + ["core-single", "core-blend", "exploratory-single", "exploratory-blend"]
PROTOCOL = Path("docs/v9-backend-protocol.json")
SNAPSHOT = Path("experiments/results/v9-similar-snapshot")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []

    def check(self, condition, context):
        self.checks += 1
        if not condition:
            self.failures.append(context)

    def close(self, actual, expected, context):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.check(actual.shape == expected.shape and
                   np.allclose(actual, expected, rtol=1e-10, atol=1e-9), context)

    def hashes(self, mapping, root=Path(".")):
        for name, digest in mapping.items():
            path = root / name
            self.check(path.is_file() and sha(path) == digest, f"sha256:{path}")


def metrics(target, prediction):
    error = (target - prediction) ** 2
    tail, maximum = target >= 1000, int(np.argmax(target))
    return dict(mse=float(error.mean()),
                mse_without_max_diagnostic_only=float(np.delete(error, maximum).mean()),
                tail_mse=float(error[tail].mean()), non_tail_mse=float(error[~tail].mean()),
                max_label_prediction=float(prediction[maximum]),
                prediction_max=float(prediction.max()), prediction_mean=float(prediction.mean()))


def select(predictions, target):
    # Independent exhaustive half-simplex, including all single-model vertices.
    grid = np.array([weights for weights in product((2, 1, 0), repeat=predictions.shape[1])
                     if sum(weights) == 2], dtype=float) / 2
    single = ((predictions - target[:, None]) ** 2).mean(axis=0)
    scores = ((predictions @ grid.T - target[:, None]) ** 2).mean(axis=0)
    return dict(single_index=int(single.argmin()), weights=grid[scores.argmin()].tolist(),
                single_mse=single.tolist(), grid_mse=scores.tolist())


def partition(audit, train, valid, universe, groups, context):
    audit.check(len(train) == len(set(train)) and len(valid) == len(set(valid)), context + ":unique")
    audit.check(bool(train) and bool(valid) and not set(train) & set(valid), context + ":disjoint")
    audit.check(set(train) | set(valid) == set(universe), context + ":coverage")
    audit.check(not set(groups[train]) & set(groups[valid]), context + ":groups")


def independent_components(identities, scaffolds):
    # Graph flood-fill, independently of the runner's union-find implementation.
    n = len(identities)
    adjacency = identities[:, None] == identities[None, :]
    adjacency |= ((scaffolds[:, None] == scaffolds[None, :]) & (scaffolds[:, None] != ""))
    result = np.full(n, -1)
    for root in range(n):
        if result[root] >= 0:
            continue
        pending = [root]
        result[root] = root
        while pending:
            node = pending.pop()
            for neighbor in np.flatnonzero(adjacency[node] & (result < 0)):
                result[neighbor] = root
                pending.append(neighbor)
    return result


def source_pair(value):
    value = str(value)
    if value.startswith("CHEMBL_ACT:"):
        matches = re.findall(r"CHEMBL_ACT:(\d+)", value)
    elif "activity " in value:
        matches = re.findall(r"activity (\d+)", value)
    elif re.fullmatch(r"CHEMBL\d+:\d+:\d+", value):
        matches = value.split(":")[1:]
    elif re.fullmatch(r"\d+/\d+", value):
        matches = value.split("/")
    else:
        return None
    return "/".join(matches) if len(matches) == 2 else None


def external_checks(audit, train, test, protocol, manifest):
    # These are the only project helpers imported: stateless structure lookups.
    from vacv.curate import standardize
    from vacv.pipeline import scaffolds_of

    external = pd.read_csv(SNAPSHOT / "provisional-exact.csv")
    raw = pd.read_csv(SNAPSHOT / "source.csv")
    info = read_json(SNAPSHOT / "manifest.json")
    audit.check(sha(SNAPSHOT / "source.csv") == info["source_sha256"], "external:source hash")
    audit.check(sha(SNAPSHOT / "provisional-exact.csv") == info["snapshot_sha256"], "external:snapshot hash")
    audit.check(len(raw) == info["source_rows"] and len(external) == info["provisional_rows"],
                "external:manifest counts")
    rows = external.source_row.to_numpy()
    audit.check(np.issubdtype(rows.dtype, np.integer) and len(set(rows)) == len(rows)
                and ((rows >= 0) & (rows < len(raw))).all(), "external:source row indices")
    selected = raw.iloc[rows].reset_index(drop=True)
    for destination, source in [("SMILES", "smiles"), ("source", "source"), ("doi", "doi")]:
        audit.check(external[destination].fillna("").tolist() == selected[source].fillna("").tolist(),
                    "external:source " + destination)
    audit.check((selected.ic50_relation == "=").all() and (selected.cc50_relation == "=").all(),
                "external:exact measurement relations")
    measurements = selected[["ic50_um", "cc50_um"]].to_numpy()
    audit.check(np.isfinite(measurements).all() and (measurements > 0).all(), "external:positive finite measurements")
    audit.close(external.target, selected.cc50_um / selected.ic50_um, "external:raw source ratio")
    audit.check(external.record_pair.tolist() == selected.source_id.map(source_pair).tolist(), "external:record pair provenance")
    audit.check(external.record_pair.notna().all() and external.record_pair.is_unique, "external:unique record pairs")
    audit.check(selected.source.str.startswith("ChEMBL").all(), "external:source family")

    identities, rejections = {}, []
    for kind, frame in [("train", train), ("test", test), ("external", external)]:
        values = []
        for row, smiles in enumerate(frame.SMILES):
            identity = standardize(smiles)
            values.append(identity.inchikey[:14] if identity.inchikey else None)
            if not identity.inchikey:
                rejections.append(dict(dataset=kind, row=row, reason=identity.reject))
        identities[kind] = np.asarray(values, dtype=object)
    audit.check(all(item["dataset"] == "test" for item in rejections), "identities:train and external valid")
    audit.check(len(rejections) == manifest["test_identity_standardization_rejections"] == 1,
                "identities:explicit single test standardization rejection")
    audit.check(identities["train"].tolist() == protocol["identities"], "identities:frozen train keys")
    audit.check(identities["external"].tolist() == external.key.tolist(), "external:standardized snapshot keys")
    blocked = (set(identities["train"]) | set(identities["test"])) - {None}
    audit.check(not set(identities["external"]) & blocked, "external:no competition standardized overlap")
    audit.check(not set(external.SMILES) & (set(train.SMILES) | set(test.SMILES)),
                "external:no exact competition SMILES overlap including rejected identity")
    audit.check(external.key.nunique() == info["provisional_identities"], "external:identity count")
    audit.check(np.isfinite(external.weight).all() and (external.weight > 0).all(), "external:finite positive weights")
    audit.close(external.groupby("key").weight.sum(),
                np.full(external.key.nunique(), .25), "external:total weight per identity")
    audit.close(external.weight, .25 / external.groupby("key").key.transform("size"), "external:per-row weights")
    train_scaffolds = np.asarray(scaffolds_of(train))
    external_scaffolds = np.asarray(scaffolds_of(external))
    components = independent_components(identities["train"], train_scaffolds)
    audit.check(components.tolist() == protocol["scaffold_groups"], "groups:independent scaffold connected components")
    details = dict(rows=len(external), identities=external.key.nunique(),
                   source_rows=len(raw), total_weight=float(external.weight.sum()),
                   standardization_rejections=rejections,
                   biological_conditions_validated=False,
                   limitation="Numerical provenance only; source endpoint, cell and exposure compatibility remain unverified")
    return external, train_scaffolds, external_scaffolds, details


def audit_artifacts(audit, output, v8, expected):
    protocol, manifest = read_json(PROTOCOL), read_json(output / "manifest.json")
    audit.check(sha(PROTOCOL) == expected == manifest["protocol_sha256"], "protocol:frozen SHA256")
    audit.check(sha(output / "protocol.json") == expected, "protocol:artifact copy")
    audit.hashes(protocol["sha256"])
    audit.hashes(manifest["sha256"], output)
    config = protocol["config"]
    audit.check(config["models"] == MODELS and config["core_candidates"] == MODELS[:-1]
                and config["exploratory_candidates"] == MODELS and config["grid_units"] == 2,
                "protocol:fixed candidates and half simplex")
    audit.check(list(protocol["folds"]) == SCHEMES, "protocol:four schemes")
    audit.check(protocol["folds"] == read_json("docs/v8-nested-protocol.json")["folds"], "protocol:V8 fold parity")
    audit.check(manifest["external_candidate_promotable"] is False
                and manifest["test_predictions"] is False and manifest["uploaded"] is False,
                "manifest:exploratory only and no test inference or upload")
    train = pd.read_csv("data/raw_v2/train_v2.csv", usecols=["SMILES", "SI VACV"])
    test = pd.read_csv("data/raw_v2/test.csv", usecols=["SMILES"])
    target = train["SI VACV"].to_numpy()
    audit.check(np.isfinite(target).all() and (target >= 0).all(), "targets:all official rows finite nonnegative")
    external, scaffolds, ext_scaffolds, external_details = external_checks(audit, train, test, protocol, manifest)
    saved_selections = read_json(output / "selection-audit.json")
    selections = {(entry["scheme"], entry["fold"]): entry for entry in saved_selections}
    audit.check(len(saved_selections) == len(selections) == 20, "selections:20 unique outer folds")
    saved_metrics = read_json(output / "metrics.json")
    metric_map = {(entry["split"], entry["model"]): entry for entry in saved_metrics}
    audit.check(len(saved_metrics) == len(metric_map) == 48, "metrics:48 unique outer rows")
    fit_audit = read_json(output / "external-fit-audit.json")
    audit.check(len(fit_audit) == 80, "external:80 fit audit entries")
    expected_artifacts = {"protocol.json", "selection-audit.json", "external-fit-audit.json", "metrics.json"}
    expected_artifacts |= {f"oof-{s}.csv" for s in SCHEMES}
    expected_artifacts |= {f"inner-{s}-{f}.csv" for s in SCHEMES for f in range(5)}
    audit.check(set(manifest["sha256"]) == expected_artifacts, "manifest:complete expected artifact inventory")
    v8_manifest = read_json(v8 / "manifest.json")
    parity, recomputed_metrics, recomputed_selections, fit_cursor = {}, [], [], 0
    for scheme in SCHEMES:
        outer = pd.read_csv(output / f"oof-{scheme}.csv")
        audit.check(outer.columns.tolist() == ["row", "target", "fold", *NAMES], scheme + ":outer schema")
        audit.check(outer.row.tolist() == list(range(len(target))), scheme + ":outer row coverage and order")
        audit.close(outer.target, target, scheme + ":outer targets")
        values = outer[NAMES].to_numpy()
        audit.check(np.isfinite(values).all() and (values >= 0).all(), scheme + ":valid outer predictions")
        old_path = v8 / f"oof-{scheme}.csv"
        audit.check(sha(old_path) == v8_manifest["artifacts_sha256"][old_path.name], scheme + ":V8 artifact hash")
        old = pd.read_csv(old_path)
        audit.check(old.row.tolist() == outer.row.tolist() and old.fold.tolist() == outer.fold.tolist(), scheme + ":V8 row/fold parity")
        audit.close(old.target, target, scheme + ":V8 target parity")
        audit.close(outer.v4, old.v4, scheme + ":V8 v4 predictions parity")
        parity[scheme] = float(np.max(np.abs(outer.v4 - old.v4)))
        groups = np.asarray(protocol["scaffold_groups"] if scheme == "scaffold" else protocol["identities"])
        seen = np.zeros(len(target), dtype=int)
        audit.check(len(protocol["folds"][scheme]) == 5, scheme + ":five outer folds")
        for fold, indices in enumerate(protocol["folds"][scheme]):
            label = f"{scheme}:{fold}"
            tr, va = indices["train"], indices["valid"]
            partition(audit, tr, va, range(len(target)), groups, label)
            seen[va] += 1
            audit.check((outer.iloc[va].fold == fold).all(), label + ":outer assignments")
            inner_seen = np.zeros(len(target), dtype=int)
            audit.check(len(indices["inner"]) == 3, label + ":three inner folds")
            for inner in indices["inner"]:
                partition(audit, inner["train"], inner["valid"], tr, groups, label + ":inner")
                inner_seen[inner["valid"]] += 1
            audit.check((inner_seen[tr] == 1).all() and (inner_seen[va] == 0).all(), label + ":inner coverage")
            inner_oof = pd.read_csv(output / f"inner-{scheme}-{fold}.csv")
            audit.check(inner_oof.columns.tolist() == ["row", "target", *MODELS], label + ":inner schema")
            audit.check(inner_oof.row.tolist() == tr, label + ":inner row order")
            audit.close(inner_oof.target, target[tr], label + ":inner targets")
            inner_predictions = inner_oof[MODELS].to_numpy()
            audit.check(np.isfinite(inner_predictions).all() and (inner_predictions >= 0).all(), label + ":valid inner predictions")
            selection = dict(scheme=scheme, fold=fold)
            for family, width, grid_size in [("core", 7, 28), ("exploratory", 8, 36)]:
                chosen = select(inner_predictions[:, :width], target[tr])
                selection[family] = chosen
                audit.check(len(chosen["grid_mse"]) == grid_size, label + ":" + family + ":grid count")
                saved = selections[scheme, fold][family]
                for key in chosen:
                    audit.close(saved[key], chosen[key], label + ":" + family + ":" + key)
                base = outer.iloc[va][MODELS[:width]].to_numpy()
                audit.close(outer.iloc[va][family + "-single"], base[:, chosen["single_index"]], label + ":" + family + ":single reconstruction")
                audit.close(outer.iloc[va][family + "-blend"], base @ chosen["weights"], label + ":" + family + ":blend reconstruction")
            recomputed_selections.append(selection)
            for split in [*indices["inner"], indices]:
                fit_entry = fit_audit[fit_cursor]
                fit_cursor += 1
                audit.check(fit_entry["scheme"] == scheme and fit_entry["fold"] == fold
                            and fit_entry["fit_rows"] == split["train"]
                            and fit_entry["query_rows"] == split["valid"], label + ":external fit/query partition")
                blocked = ((set(scaffolds[va]) | set(scaffolds[split["valid"]])) - {""}
                           if scheme == "scaffold" else set())
                expected_rows = [i for i, value in enumerate(ext_scaffolds) if value not in blocked]
                audit.check(fit_entry["external_rows"] == expected_rows, label + ":external eligible rows and scaffold isolation")
                remaining_weights = external.iloc[expected_rows].groupby("key").weight.sum()
                audit.check((remaining_weights <= .25 + 1e-12).all(), label + ":external weight ceiling")
        audit.check((seen == 1).all(), scheme + ":outer exactly-once coverage")
        for model in NAMES:
            recomputed = dict(split=scheme, model=model, **metrics(target, outer[model].to_numpy()))
            for key, value in recomputed.items():
                if key not in ("split", "model"):
                    audit.close(metric_map[scheme, model][key], value, scheme + ":" + model + ":metric:" + key)
            recomputed_metrics.append(recomputed)
    scores = {(m["split"], m["model"]): m["mse"] for m in recomputed_metrics}
    confirmed = (all(scores[s, "core-blend"] < scores[s, "v4"] for s in SCHEMES[:3])
                 and scores["scaffold", "core-blend"] <= scores["scaffold", "v4"])
    audit.check(manifest["confirmation_passed"] == confirmed, "confirmation:independent declared rule")
    audit.check(fit_cursor == 80, "external:all fit records consumed")
    return dict(protocol_sha256=expected, manifest_sha256=sha(output / "manifest.json"),
                training_rows=len(target), outer_metrics_recomputed=len(recomputed_metrics),
                core_selections_recomputed=len(recomputed_selections),
                exploratory_selections_recomputed=len(recomputed_selections),
                external_fit_records_checked=fit_cursor, v4_v8_max_abs_difference=parity,
                confirmation_passed=confirmed, external=external_details,
                metrics=recomputed_metrics, selections=recomputed_selections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/v9-backend-validation"))
    parser.add_argument("--v8-dir", type=Path, default=Path("experiments/results/v8-nested-validation"))
    parser.add_argument("--protocol-sha256", default=EXPECTED_PROTOCOL)
    args = parser.parse_args()
    if not (args.output_dir / "manifest.json").is_file():
        print("Not ready: completed V9 manifest does not exist; no audit report written.")
        return 2
    audit, report = Audit(), {}
    try:
        report = audit_artifacts(audit, args.output_dir, args.v8_dir, args.protocol_sha256)
    except Exception as error:
        audit.failures.append(f"Audit stopped: {type(error).__name__}: {error}")
    report.update(passed=not audit.failures, checks=audit.checks, failures=audit.failures,
                  audited_at=datetime.now(timezone.utc).isoformat(), audit_source_sha256=sha(__file__),
                  method="Saved-artifact recomputation; no model fitting or fitting-code imports")
    destination = args.output_dir / "independent-audit.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "checks", "failures")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
