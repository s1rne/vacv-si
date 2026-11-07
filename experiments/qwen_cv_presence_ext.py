"""CV на патченой presence-таблице (Задача 4).

Копия логики scripts/cv_checkpoint.py с одной заменой: presence-признаки
подтягиваются из файла, указанного --presence (monkey-patch
vacv.literature.presence; vacv/ не редактируется). Прогон идентичен
checkpoint-av: те же сплиты/фолды/модель, включая хард-оверрайды
apply_article_overrides внутри predict_components — дельта между прогонами
объясняется только изменением presence-признаков.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import vacv.literature as literature
from vacv.checkpoint import BestCheckpointModel, load_embeddings
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--presence", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--features", type=Path, default=Path("data/raw/models/chemeleon_features.npz"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/raw/models/tabicl-classifier-v2-20260212.ckpt"))
    parser.add_argument("--splits", nargs="+", default=["3"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # подмена источника presence-признаков
    patched_path = args.presence
    literature.presence = lambda path=None: pd.read_csv(
        patched_path, low_memory=False,
        usecols=["smiles", *literature.PRESENCE_FEATURES]
    ).drop_duplicates("smiles").rename(columns={"smiles": "SMILES"})

    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use an empty output directory")
    data = load()
    frame, test, y = literature.attach(data.train), literature.attach(data.test), data.y
    embeddings, _ = load_embeddings(args.features)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for split in args.splits:
        folds = (scaffold_folds(scaffolds_of(frame), 5) if split == "scaffold" else
                 [v for _, v in KFold(5, shuffle=True, random_state=int(split)).split(y)])
        prediction = np.full(len(y), np.nan)
        probability = np.full(len(y), np.nan)
        fold_id = np.full(len(y), -1)
        for fold in range(5):
            train, valid = fold_indices(folds, fold)
            model = BestCheckpointModel(embeddings, args.checkpoint, active_value=True).fit(
                frame.iloc[train], y[train])
            components = model.predict_components(frame.iloc[valid], unlabeled=test)
            prediction[valid], probability[valid] = components["prediction"], components["probability"]
            fold_id[valid] = fold
            print(f"tag={args.tag} split={split} fold={fold} complete", flush=True)
        if not np.isfinite(prediction).all():
            raise ValueError("Incomplete OOF predictions")
        row = {"split": split, "tag": args.tag, "mse": float(np.mean((prediction - y) ** 2)),
               "model": "checkpoint-av"}
        results.append(row)
        pd.DataFrame({"index": np.arange(len(y)), "fold": fold_id, "probability": probability,
                      "prediction": prediction}).to_csv(args.out / f"oof-{split}.csv", index=False)
        (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
