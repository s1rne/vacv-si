"""SI из двух голов против текущего блендинга на выборке соревнования.

Головы обучены на внешнем корпусе и целевой переменной соревнования не видят вовсе,
поэтому первые два блока печатаются без кросс-проверки: это честный перенос.
Кросс-проверка нужна только там, где по обучающему фолду подбираются базовая частота
заглушки и калибровка шкалы отношения.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from vacv.dataset import load
from vacv.endpoints import load_corpus
from vacv.heads import EndpointHeads
from vacv.literature import apply, attach
from vacv.multitask import MultitaskModel
from vacv.pipeline import INACTIVE, SelectivityModel, scaffolds_of
from vacv.splits import fold_indices, scaffold_folds


def competition_splits(y: np.ndarray, frame: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    return {"случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(y)],
            "остовы": scaffold_folds(scaffolds_of(frame), n_splits=5)}


def blend_oof(frame: pd.DataFrame, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    prediction = np.zeros_like(y)
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        model = SelectivityModel().fit(frame.iloc[train_index], y[train_index])
        prediction[validation_index] = model.predict(frame.iloc[validation_index])
    return prediction


def multitask_oof(heads: EndpointHeads, frame: pd.DataFrame, y: np.ndarray,
                  folds: list[np.ndarray], **kwargs) -> np.ndarray:
    prediction = np.zeros_like(y)
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        model = MultitaskModel(heads, **kwargs).fit(frame.iloc[train_index], y[train_index])
        prediction[validation_index] = model.predict(frame.iloc[validation_index])
    return prediction


def main() -> None:
    data = load()
    y = data.y
    frame = attach(data.train)
    labels = (y == INACTIVE).astype(int)
    corpus = load_corpus()
    heads = EndpointHeads().fit(corpus)
    predicted = heads.predict(data.train)

    print("перенос головы неактивности на заглушку соревнования")
    print(f"  неактивных в корпусе {heads.prior_:.0%}, заглушек в выборке {labels.mean():.0%}")
    print(f"  AUC по 320 строкам обучающей выборки: {roc_auc_score(labels, predicted['p_inactive']):.3f}")
    print(f"  средняя p у заглушек {predicted.loc[labels == 1, 'p_inactive'].mean():.3f}, "
          f"у активных {predicted.loc[labels == 0, 'p_inactive'].mean():.3f}")

    active = labels == 0
    log_y = np.log10(np.clip(y[active], 1e-3, None))
    log_si = predicted.loc[active, "log_si"].to_numpy(float)
    print("\nпроизводный log10 SI на активных строках")
    print(f"  Spearman {spearmanr(log_si, log_y).statistic:.3f}, "
          f"Pearson {np.corrcoef(log_si, log_y)[0, 1]:.3f}")
    print(f"  RMSE без калибровки {np.sqrt(np.mean((log_si - log_y) ** 2)):.3f}, "
          f"сдвиг {np.mean(log_y - log_si):+.3f}")
    design = np.column_stack([np.ones_like(log_si), log_si])
    fitted = design @ np.linalg.lstsq(design, log_y, rcond=None)[0]
    print(f"  RMSE после линейной калибровки {np.sqrt(np.mean((fitted - log_y) ** 2)):.3f} "
          f"(разброс метки {log_y.std():.3f}, шумовой предел 0.96)")

    print("\nMSE на исходной шкале SI")
    print(f"{'модель':34} " + " ".join(f"{name:>14}" for name in competition_splits(y, data.train)))
    rows: dict[str, dict[str, float]] = {}
    for name, folds in competition_splits(y, data.train).items():
        blend = blend_oof(frame, y, folds)
        variants = {
            "константа": np.full_like(y, y.mean()),
            "блендинг (текущий)": blend,
            "блендинг + литература": apply(blend, frame),
            "головы, средний SI из голов": multitask_oof(heads, data.train, y, folds),
            "головы, средний SI константой": multitask_oof(heads, data.train, y, folds, active="constant"),
            "головы без сдвига частоты": multitask_oof(heads, data.train, y, folds, shift=False),
        }
        variants["головы + литература"] = apply(variants["головы, средний SI из голов"], frame)
        for label, prediction in variants.items():
            rows.setdefault(label, {})[name] = float(np.mean((prediction - y) ** 2))
    for label, scores in rows.items():
        print(f"{label:34} " + " ".join(f"{value:14,.0f}" for value in scores.values()))


if __name__ == "__main__":
    main()
