"""A/B честного CV: presence базовая (с MCC) vs +ID50-как-antiviral (верхняя
граница эффекта, experiments/qwen_id50_av_patch_build.py →
outputs/qwen-sentinel/presence_id50av.csv, 34 строки: 17 train / 17 test).

Протокол как cv_mcc_ab.py: правило без y (только ChEMBL-записи), модель в
каждом фолде учится на fit=4/5, оценка на valid=1/5.

Запуск: uv run python experiments/cv_id50_ab.py --splits 3 4 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import vacv.literature as literature
from sklearn.model_selection import KFold

from vacv.checkpoint import BestCheckpointModel, load_embeddings
from vacv.dataset import load
from vacv.literature import attach
from vacv.pipeline import scaffolds_of
from vacv.splits import fold_indices, scaffold_folds

EXT = Path("outputs/qwen-sentinel/presence_id50av.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["3", "4", "5"])
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("data/raw/models/tabicl-classifier-v2-20260212.ckpt"))
    ap.add_argument("--features", type=Path, default=Path("data/raw/models/chemeleon_features.npz"))
    ap.add_argument("--out", type=Path, default=Path("outputs/doc-scraper/id50-ab"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = load()
    y = np.asarray(data.y, dtype=float)
    embeddings, _ = load_embeddings(args.features)

    rows = []
    for tag, path in (("base", literature.PRESENCE), ("id50", EXT)):
        # presence() держит путь в default-аргументе — подменяем его
        literature.presence.__defaults__ = (path,)
        frame, test = attach(data.train), attach(data.test)
        for split in args.splits:
            folds = (scaffold_folds(scaffolds_of(frame), 5) if split == "scaffold" else
                     [v for _, v in KFold(5, shuffle=True, random_state=int(split)).split(y)])
            prediction = np.full(len(y), np.nan)
            for fold in range(5):
                train, valid = fold_indices(folds, fold)
                model = BestCheckpointModel(embeddings, args.checkpoint, active_value=True) \
                    .fit(frame.iloc[train], y[train])
                prediction[valid] = model.predict_components(
                    frame.iloc[valid], unlabeled=test)["prediction"]
                print(f"{tag} {split} fold{fold} done", flush=True)
            mse = float(np.mean((prediction - y) ** 2))
            pd.DataFrame({"index": np.arange(len(y)), "prediction": prediction}) \
                .to_csv(args.out / f"oof-{tag}-{split}.csv", index=False)
            rows.append({"tag": tag, "split": split, "mse": mse})
            print(json.dumps({"tag": tag, "split": split, "mse": mse}), flush=True)

    res = pd.DataFrame(rows)
    piv = res.pivot(index="split", columns="tag", values="mse")
    piv["delta"] = piv["id50"] - piv["base"]
    print(piv.to_string(float_format=lambda v: f"{v:,.0f}"))
    print(f"средний эффект: {(piv['id50'].mean() - piv['base'].mean()) / piv['base'].mean():+.2%}")
    res.to_csv(args.out / "metrics.csv", index=False)
    print(f"→ {args.out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
