"""Link digitised literature/patent activity tables to competition rows.

Reads `data/raw/literature_tables/sources.json`, where each entry is one
published activity table transcribed verbatim, and links its rows to the
competition compounds by the paper's own compound label, inside the
`strain` + `Culture` groups the table can belong to and inside the structural
series the table is about.

Validation uses only the training split, whose CC50/IC50 are visible, and the
report is printed per source. A source that does not reproduce the training
numbers is not written out.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/raw/literature_tables/sources.json"
OUT = ROOT / "data/processed/literature_manual.csv"

GROUPS = {
    "G1": "Vaccinia virus (strain Copenhagen). culture Vero",
    "G2": (
        "Vaccinia virus (strain Copenhagen). culture Vero, cowpox virus "
        "(strain_Grishak), ectromelia virus (ECTV, strain K-1)"
    ),
    "G3": (
        "Vaccinia virus (strain Copenhagen). culture Vero, cowpox virus "
        "(strain_Grishak), ectromelia (strain K-1)"
    ),
    "G4": "Vaccinia virus, lung fibroblast cells",
    "G5": "Vaccinia virus (strain Copenhagen). culture Vero,",
    "G6": (
        "Vaccinia virus (strain Copenhagen). culture Vero, cowpox virus "
        "(strain_Grishak)"
    ),
}

REGISTRY = re.compile(r"^(?P<label>[0-9]+[a-z]?'?)-(?:[a-z]{2}|qc|qs|ks|as|ya)-\d+")


def competition() -> pd.DataFrame:
    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(ROOT / f"data/raw/kaggle/{split}.csv")
        df["split"] = split
        df["index"] = range(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def label_forms(title: str) -> list[str]:
    """Every plausible paper label hiding in a competition `Title`."""
    title = str(title).strip()
    forms = [title]
    head = title.split(" (")[0].strip()
    forms.append(head)
    m = REGISTRY.match(head)
    if m:
        forms.append(m.group("label"))
    inner = re.findall(r"\(([^)]*)\)", title)
    forms.extend(x.strip() for x in inner)
    out = []
    for f in forms:
        f = f.strip().lower().replace("’", "'").replace("`", "'")
        if f and f not in out:
            out.append(f)
    return out


def number(value: str) -> tuple[float | None, str]:
    """Parse a printed cell into (value, relation)."""
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NT", "N/A", "-", "NAN"}:
        return None, ""
    relation = ""
    if text[0] in "><":
        relation, text = text[0], text[1:]
    text = text.replace(" ", "").replace(",", ".").replace(" ", "")
    try:
        return float(text), relation
    except ValueError:
        return None, relation


def matches(mol, smarts: list[str], not_smarts: list[str] = ()) -> bool:
    for pattern in smarts:
        query = Chem.MolFromSmarts(pattern)
        if query is None:
            raise SystemExit(f"bad SMARTS: {pattern}")
        if not mol.HasSubstructMatch(query):
            return False
    for pattern in not_smarts:
        query = Chem.MolFromSmarts(pattern)
        if query is None:
            raise SystemExit(f"bad SMARTS: {pattern}")
        if mol.HasSubstructMatch(query):
            return False
    return True


def series_filter(source: dict, label: str) -> tuple[list[str], list[str]]:
    """SMARTS for the structural series the label belongs to."""
    label = label.strip().lower()
    for entry in source.get("series", []):
        if "labels" in entry:
            if label in [x.lower() for x in entry["labels"]]:
                return entry.get("smarts", []), entry.get("not_smarts", [])
        elif label.startswith(entry["prefix"]):
            return entry.get("smarts", []), entry.get("not_smarts", [])
    return source.get("smarts", []), source.get("not_smarts", [])


def candidates(comp: pd.DataFrame, source: dict) -> pd.DataFrame:
    strains = [GROUPS[g] for g in source["groups"]]
    return comp[comp["strain"].isin(strains)].copy()


def close(a: float | None, b: float | None, tol: float = 0.02) -> bool:
    if a is None or b is None or (isinstance(b, float) and math.isnan(b)):
        return a is None and (b is None or (isinstance(b, float) and math.isnan(b)))
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def link(comp: pd.DataFrame, source: dict) -> tuple[list[dict], dict]:
    pool = candidates(comp, source)
    index: dict[str, list[int]] = {}
    for row_id, row in pool.iterrows():
        for form in label_forms(row["Title"]):
            index.setdefault(form, []).append(row_id)
    explicit = source.get("explicit", {})
    position = {(r["split"], int(r["index"])): i for i, r in pool.iterrows()}

    linked: list[dict] = []
    report = {
        "id": source["id"],
        "candidates": len(pool),
        "table_rows": len(source["rows"]),
        "ambiguous": 0,
        "unmatched": 0,
        "train": 0,
        "cc50_identical": 0,
        "cc50_compared": 0,
        "ic50_identical": 0,
        "ic50_compared": 0,
        "mismatch": [],
    }

    for entry in source["rows"]:
        key = entry["label"].strip().lower()
        if entry["label"] in explicit:
            hits = [position[(split, idx)] for split, idx in explicit[entry["label"]]]
        else:
            hits = index.get(key, [])
            smarts, not_smarts = series_filter(source, entry["label"])
            if smarts or not_smarts:
                kept = []
                for row_id in hits:
                    mol = Chem.MolFromSmiles(str(pool.loc[row_id, "SMILES"]))
                    if mol and matches(mol, smarts, not_smarts):
                        kept.append(row_id)
                hits = kept
        if not hits:
            report["unmatched"] += 1
            continue
        cc50, cc_rel = number(entry.get("cc50", ""))
        ic50, ic_rel = number(entry.get("ic50", ""))
        si, _ = number(entry.get("si", ""))
        for row_id in hits:
            row = pool.loc[row_id]
            if len(hits) > 1:
                report["ambiguous"] += 1
            if row["split"] == "train":
                report["train"] += 1
                own_cc = row.get("CC50_uM")
                own_ic = row.get("IC50_uM_VACV")
                if cc50 is not None:
                    report["cc50_compared"] += 1
                    if close(cc50, own_cc):
                        report["cc50_identical"] += 1
                    else:
                        report["mismatch"].append(
                            f"{entry['label']}: CC50 printed {entry['cc50']} vs competition {own_cc}"
                        )
                report["ic50_compared"] += 1
                if close(ic50, None if pd.isna(own_ic) else float(own_ic)):
                    report["ic50_identical"] += 1
                else:
                    report["mismatch"].append(
                        f"{entry['label']}: IC50 printed {entry['ic50'] or 'NA'} vs competition {own_ic}"
                    )
            linked.append(
                {
                    "split": row["split"],
                    "index": int(row["index"]),
                    "Title": row["Title"],
                    "SMILES": row["SMILES"],
                    "paper_doi_or_url": source["doi"],
                    "paper_citation": source["citation"],
                    "paper_label": entry["label"],
                    "ic50": f"{ic_rel}{ic50:g}" if ic50 is not None else "",
                    "cc50": f"{cc_rel}{cc50:g}" if cc50 is not None else "",
                    "si": f"{si:g}" if si is not None else "",
                    "units_original": source["units_original"],
                    "cell_line": source["cell_line"],
                    "strain": source["strain"],
                    "link_key": "label within the strain group, series-filtered",
                    "confidence": "high" if len(hits) == 1 else "medium",
                }
            )
    return linked, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="append to literature_manual.csv")
    parser.add_argument("--only", help="run a single source id")
    args = parser.parse_args()

    comp = competition()
    sources = json.loads(SOURCES.read_text())
    if args.only:
        sources = [s for s in sources if s["id"] == args.only]

    all_rows: list[dict] = []
    for source in sources:
        linked, report = link(comp, source)
        print(f"== {report['id']}  {source['doi']}")
        print(
            f"   candidate competition rows {report['candidates']}, "
            f"table rows {report['table_rows']}, linked {len(linked)}, "
            f"unmatched labels {report['unmatched']}, ambiguous links {report['ambiguous']}"
        )
        print(
            f"   train rows hit {report['train']}: "
            f"CC50 identical {report['cc50_identical']}/{report['cc50_compared']}, "
            f"IC50 identical {report['ic50_identical']}/{report['ic50_compared']}"
        )
        for line in report["mismatch"]:
            print(f"   ! {line}")
        test_new = sum(1 for r in linked if r["split"] == "test")
        print(f"   test rows covered {test_new}")
        all_rows.extend(linked)

    if not args.write:
        return 0

    new = pd.DataFrame(all_rows)
    existing = pd.read_csv(OUT, dtype=str)
    new = new[existing.columns]
    merged = pd.concat([existing, new.astype(str)], ignore_index=True)
    merged = merged.drop_duplicates(subset=["split", "index", "paper_doi_or_url"], keep="first")
    merged.to_csv(OUT, index=False)
    print(f"\nwrote {len(merged)} rows to {OUT} ({len(merged) - len(existing)} new)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
