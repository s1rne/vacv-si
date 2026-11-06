"""Состязательная валидация: отличимы ли строки теста от строк обучения.

Если классификатор train-vs-test уверенно разделяет выборки, оценка на обучении
смещена, и видно, какие признаки за это отвечают.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import frames
from vacv.literature import PRESENCE_FEATURES
from vacv.pipeline import ContextEncoder, structure_matrix, title_matrix


def blocks(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    both = pd.concat([train, test], ignore_index=True)
    encoder = ContextEncoder().fit(both)
    context = np.hstack([
        pd.get_dummies(both[name].astype("category"), prefix=name).to_numpy(float)
        for name in ("strain", "Culture")
    ])
    return {
        "структура (Morgan+MACCS)": structure_matrix(both),
        "контекст (strain+Culture)": context,
        "обозначение (Title)": title_matrix(both["Title"]),
        "открытые базы": both[["matched", *PRESENCE_FEATURES]].to_numpy(float, na_value=-1.0),
        "всё вместе": encoder.transform(both),
    }


def auc(x: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    out = np.zeros(len(y))
    for fit, check in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed)
        out[check] = model.fit(x[fit], y[fit]).predict_proba(x[check])[:, 1]
    return roc_auc_score(y, out)


def main() -> None:
    train, test, _ = frames()
    label = np.r_[np.zeros(len(train)), np.ones(len(test))]
    print(f"{'набор признаков':28} {'AUC train-vs-test':>18}")
    for name, matrix in blocks(train, test).items():
        print(f"{name:28} {auc(matrix, label):18.3f}")

    both = pd.concat([train, test], ignore_index=True)
    columns = ["matched", *PRESENCE_FEATURES]
    named = np.hstack([
        both[columns].to_numpy(float, na_value=-1.0),
        title_matrix(both["Title"]),
        pd.get_dummies(both["strain"].astype("category"), prefix="strain").to_numpy(float),
        pd.get_dummies(both["Culture"].astype("category"), prefix="culture").to_numpy(float),
    ])
    names = columns + ["title_len", "title_digits", "title_is_number", "title_has_dash", "title_leading"] \
        + [f"strain={v}" for v in sorted(both["strain"].astype("category").cat.categories)] \
        + [f"culture={v}" for v in sorted(both["Culture"].astype("category").cat.categories)]
    model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0)
    model.fit(named, label)
    importance = permutation_importance(model, named, label, n_repeats=10, random_state=0, scoring="roc_auc")
    order = np.argsort(-importance.importances_mean)[:12]
    print("\nчто разделяет выборки (перестановочная важность по AUC):")
    for position in order:
        print(f"  {names[position]:34} {importance.importances_mean[position]:+.4f}")

    print("\nсдвиг распределений по группам strain+Culture:")
    group = lambda f: f[["strain", "Culture"]].astype(str).agg(" | ".join, axis=1)
    counts = pd.DataFrame({
        "train": group(train).value_counts(),
        "test": group(test).value_counts(),
    }).fillna(0).astype(int)
    counts["доля train"] = counts["train"] / counts["train"].sum()
    counts["доля test"] = counts["test"] / counts["test"].sum()
    shared = counts[(counts["train"] > 0) & (counts["test"] > 0)]
    print(f"  групп всего {len(counts)}, общих {len(shared)}")
    print(f"  тестовых строк в общих группах {shared['test'].sum()} из {counts['test'].sum()} "
          f"({shared['test'].sum() / counts['test'].sum():.1%})")
    print(f"  обучающих строк в общих группах {shared['train'].sum()} из {counts['train'].sum()}")
    moved = counts.assign(delta=(counts["доля test"] - counts["доля train"]).abs()).nlargest(8, "delta")
    print(moved[["train", "test", "доля train", "доля test"]].to_string())


if __name__ == "__main__":
    main()
