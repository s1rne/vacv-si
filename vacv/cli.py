"""Command line entry point for the data layer.

    uv run python -m vacv.cli all          # fetch, curate, build, validate
    uv run python -m vacv.cli fetch        # open sources only, cached in data/raw
    uv run python -m vacv.cli fetch --source chembl --refresh
    uv run python -m vacv.cli curate
    uv run python -m vacv.cli build
    uv run python -m vacv.cli validate

Every stage writes its output to disk and the next stage reads it from there,
so a failure never costs more than one stage, and the slow part (a few hundred
ChEMBL API pages) is paid once.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import (
    build_dataset,
    competition,
    coverage as coverage_module,
    datacard,
    match_competition,
    presence,
    validate as validate_module,
)
from .curate import curate, inactive_evidence
from .paths import INACTIVE, INTERIM, KAGGLE, PROCESSED, ensure_dirs
from .schema import conform
from .sources import ALL_SOURCES, OPEN_SOURCES, kaggle_competition

log = logging.getLogger("vacv")

MEASUREMENTS = INTERIM / "measurements.csv.gz"
CURATED = INTERIM / "curated.csv.gz"
MAIN_OUTPUT = PROCESSED / "vacv_si.csv"


def _source_path(name: str) -> "object":
    return INTERIM / f"{name}.csv.gz"


def cmd_fetch(args: argparse.Namespace) -> int:
    ensure_dirs()
    INTERIM.mkdir(parents=True, exist_ok=True)
    registry = ALL_SOURCES if args.include_competition else OPEN_SOURCES
    names = [args.source] if args.source else list(registry)
    for name in names:
        if name not in ALL_SOURCES:
            log.error("unknown source %s; known: %s", name, list(ALL_SOURCES))
            return 2
        log.info("fetching %s", name)
        frame = ALL_SOURCES[name]()
        frame.to_csv(_source_path(name), index=False)
        log.info("%s -> %d measurements", name, len(frame))
    return _combine(args.include_competition)


def _combine(include_competition: bool) -> int:
    registry = ALL_SOURCES if include_competition else OPEN_SOURCES
    frames = []
    for name in registry:
        path = _source_path(name)
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        log.error("nothing fetched yet")
        return 1
    combined = conform(pd.concat(frames, ignore_index=True))
    combined = _deduplicate(combined)
    combined.to_csv(MEASUREMENTS, index=False)
    log.info("combined %d measurements -> %s", len(combined), MEASUREMENTS)
    return 0


def _deduplicate(combined: pd.DataFrame) -> pd.DataFrame:
    """Drop measurements two harvesters fetched.

    ``chembl`` and ``chembl_targeted`` query the same database from different
    directions and overlap heavily on purpose. ``source_id`` is one activity
    in one source, so the same activity arriving twice is one measurement,
    and keeping both would double its weight in every median.
    """
    identified = combined["source_id"].notna()
    duplicated = identified & combined.duplicated(subset=["source_db", "source_id"])
    if duplicated.any():
        log.info("dropping %d measurements fetched by more than one harvester", int(duplicated.sum()))
    return combined[~duplicated].reset_index(drop=True)


def cmd_curate(args: argparse.Namespace) -> int:
    if not MEASUREMENTS.exists():
        log.error("%s missing; run `fetch` first", MEASUREMENTS)
        return 1
    raw = pd.read_csv(MEASUREMENTS, low_memory=False)
    kept, reasons = curate(raw, canonical_tautomer=not args.no_tautomer)
    kept.to_csv(CURATED, index=False)
    inactive = inactive_evidence(raw, canonical_tautomer=not args.no_tautomer)
    inactive.to_csv(INACTIVE, index=False)
    print(f"{len(inactive)} inactive-against-vaccinia records -> {INACTIVE}")
    print(f"curated {len(kept)} of {len(raw)} measurements -> {CURATED}")
    for reason, count in reasons.most_common():
        print(f"  dropped {count:7d}  {reason}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    if not CURATED.exists():
        log.error("%s missing; run `curate` first", CURATED)
        return 1
    curated = pd.read_csv(CURATED, low_memory=False)
    # The deliverable is an open-source dataset. Competition rows may sit in
    # the curated table (they are needed for the overlap analysis), but they
    # never reach data/processed.
    open_only = match_competition.open_source_rows(curated)
    if len(open_only) != len(curated):
        log.info("build: excluding %d competition rows from the dataset", len(curated) - len(open_only))
    tables = build_dataset.build(open_only)
    written = build_dataset.write(tables)
    for name, path in written.items():
        print(f"wrote {path}")
    competition = build_dataset.load_competition_set()
    analysis = None
    if (KAGGLE / "test.csv").exists() or (KAGGLE / "train.csv").exists():
        analysis = match_competition.analyse(curated)
        for path in match_competition.write_lookups(analysis, PROCESSED):
            print(f"wrote {path}")
        _print_match(analysis)
        print(f"wrote {_write_presence(open_only)}")
    card = datacard.render(open_only, tables, competition, analysis)
    datacard.write(card)
    print(f"wrote {datacard.OUTPUT}")
    return 0


def _print_match(analysis: dict) -> None:
    from .match_competition import HEADER

    print(" | ".join(HEADER))
    for item in analysis["overlaps"]:
        print(" | ".join(str(c) for c in item.as_row()))
    print("agreement with the organisers' SI on train:", analysis["agreement_summary"])
    print("cell-line-aware lookup agreement:", analysis.get("lookup_agreement"))
    _print_recovery(analysis)
    documents = analysis["documents"]
    if not documents.empty:
        print(f"{len(documents)} publications cover competition compounds; top 10:")
        print(documents.head(10).to_string(index=False))


def _write_presence(curated: pd.DataFrame):
    table = presence.presence_table(curated)
    path = presence.write(table, PROCESSED)
    report = presence.sentinel_report(table)
    print("sentinel separation on train (AUC for the sentinel class):")
    print(report.to_string(index=False))
    return path


def cmd_presence(args: argparse.Namespace) -> int:
    if not CURATED.exists():
        log.error("%s missing; run `curate` first", CURATED)
        return 1
    curated = match_competition.open_source_rows(pd.read_csv(CURATED, low_memory=False))
    print(f"wrote {_write_presence(curated)}")
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    if not CURATED.exists():
        log.error("%s missing; run `curate` first", CURATED)
        return 1
    curated = match_competition.open_source_rows(pd.read_csv(CURATED, low_memory=False))
    splits = competition.splits()
    if not splits:
        log.error("no competition file in %s", KAGGLE)
        return 1
    result = match_competition.recovery(curated, splits)
    _print_recovery(result)
    path = coverage_module.write(result["recovered"], PROCESSED)
    print(f"wrote {path}")
    return 0


def _print_recovery(result: dict) -> None:
    table = result.get("recovery_coverage")
    if table is not None and not table.empty:
        print("\ncoverage of the competition set:")
        print(table.to_string(index=False))
    for name, report in (result.get("recovery_report") or {}).items():
        print(f"\nreproduction of the organisers' values on train, by {name}:")
        print(report.to_string(index=False))
    split_classes = result.get("recovery_by_sentinel")
    if split_classes is not None and not split_classes.empty:
        print("\ntrain coverage, by what the organisers' row actually is:")
        print(split_classes.to_string(index=False))
    sentinel = result.get("recovery_sentinel")
    if sentinel is not None and not sentinel.empty:
        print("\nsentinel (`SI VACV == 10000`) share on train, by evidence:")
        print(sentinel.to_string(index=False))


def cmd_match(args: argparse.Namespace) -> int:
    if not CURATED.exists():
        log.error("%s missing; run `curate` first", CURATED)
        return 1
    analysis = match_competition.analyse(pd.read_csv(CURATED, low_memory=False))
    _print_match(analysis)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = args.path or MAIN_OUTPUT
    if not path.exists():
        log.error("%s missing; run `build` first", path)
        return 1
    frame = pd.read_csv(path, low_memory=False)
    findings = validate_module.report(frame)
    return 1 if validate_module.errors(findings) else 0


def cmd_all(args: argparse.Namespace) -> int:
    for step in (cmd_fetch, cmd_curate, cmd_build, cmd_validate):
        code = step(args)
        if code:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vacv.cli", description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="query the open databases")
    fetch.add_argument("--source", choices=sorted(ALL_SOURCES))
    fetch.add_argument("--refresh", action="store_true", help="ignore the data/raw cache")
    fetch.set_defaults(func=cmd_fetch)

    curate_parser = sub.add_parser("curate", help="standardize structures and units")
    curate_parser.add_argument("--no-tautomer", action="store_true", help="skip canonical tautomer")
    curate_parser.set_defaults(func=cmd_curate)

    build = sub.add_parser("build", help="pair IC50 with CC50 and write the dataset")
    build.set_defaults(func=cmd_build)

    validate_parser = sub.add_parser("validate", help="check the built dataset")
    validate_parser.add_argument("--path", type=lambda p: PROCESSED / p if "/" not in p else p)
    validate_parser.set_defaults(func=cmd_validate)

    match = sub.add_parser("match", help="overlap with the competition set")
    match.set_defaults(func=cmd_match)

    presence_parser = sub.add_parser(
        "presence", help="open-data evidence features for every competition compound"
    )
    presence_parser.set_defaults(func=cmd_presence)

    coverage_parser = sub.add_parser(
        "coverage",
        help="recover the organisers' own IC50/CC50 per compound and score the rule on train",
    )
    coverage_parser.set_defaults(func=cmd_coverage)

    everything = sub.add_parser("all", help="fetch, curate, build, validate")
    everything.add_argument("--no-tautomer", action="store_true")
    everything.add_argument("--refresh", action="store_true")
    everything.set_defaults(func=cmd_all, source=None, path=None)

    for subparser in (fetch, everything):
        subparser.add_argument(
            "--include-competition",
            action="store_true",
            help=f"also read the organisers' CSVs from {kaggle_competition.KAGGLE}",
        )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for attribute, default in (
        ("source", None),
        ("path", None),
        ("no_tautomer", False),
        ("refresh", False),
        ("include_competition", False),
    ):
        if not hasattr(args, attribute):
            setattr(args, attribute, default)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
