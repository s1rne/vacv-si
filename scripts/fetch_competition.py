"""Скачать файлы соревнования в data/raw/kaggle.

Нужен токен Kaggle в ~/.kaggle/kaggle.json и принятые правила соревнования на сайте.
Файл обучения называется train_v2.csv, загрузчик ждёт имя train.csv, поэтому копируем.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

COMPETITION = "molecular-activity-prediction-hack"
TARGET = Path("data/raw/kaggle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=TARGET)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(args.out)], check=True)
    for archive in args.out.glob("*.zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(args.out)
        archive.unlink()

    updated = args.out / "train_v2.csv"
    if updated.exists():
        shutil.copyfile(updated, args.out / "train.csv")
    missing = [name for name in ("train.csv", "test.csv", "sample_submit.csv") if not (args.out / name).exists()]
    if missing:
        raise SystemExit(f"не хватает файлов: {missing}")
    print("готово:", ", ".join(sorted(path.name for path in args.out.glob("*.csv"))))


if __name__ == "__main__":
    main()
