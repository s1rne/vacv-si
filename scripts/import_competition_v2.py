"""Import official v2 data, excluding prohibited fields before parsing."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd


def import_archive(archive, out):
    archive, out = Path(archive), Path(out)
    if out.exists():
        raise FileExistsError("Use a new output directory; old datasets are preserved")
    allowed = {"index", "Title", "SMILES", "strain", "Culture"}
    tables, hashes = {}, {}
    with zipfile.ZipFile(archive) as source:
        for name in ("train_v2.csv", "test.csv", "sample_submit.csv"):
            fields = (allowed | {"SI VACV"} if name == "train_v2.csv" else
                      {"index", "SI VACV"} if name == "sample_submit.csv" else allowed)
            raw = source.read(name)
            hashes[name] = hashlib.sha256(raw).hexdigest()
            with source.open(name) as stream:
                tables[name] = pd.read_csv(stream, usecols=lambda c: c in fields)
    out.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    manifest = {"target_version": "train_v2", "source": "official Kaggle competition ZIP",
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "source_file_sha256": hashes,
                "sanitized_file_sha256": {name: hashlib.sha256((out/name).read_bytes()).hexdigest()
                                          for name in tables}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/raw_v2"))
    args = parser.parse_args()
    print(json.dumps(import_archive(args.archive, args.out), indent=2))
