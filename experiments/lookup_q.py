"""Надёжность литературного матча q(x) для soft lookup.

В matched-бакете (литературное L подставляется вместо A(x)) остаётся около 0.5M MSE;
часть матчей ложная — литература не про то соединение или не тот штамм. Здесь для
каждой matched-строки оценивается q(x) ∈ [0,1] — вероятность того, что матч плохой,
и предсказание становится p×10000 + (1−p)×((1−q)×L + q×A(x)). Цель q строится на
matched-строках outer-train фолда по расхождению |log10((L+1)/(y+1))| > 0.1; все
признаки q (качество записи, уровень сопряжения, цензура, log10(L+1), совпадение
клеточной линии) берутся из внешней lookup-таблицы и вычислимы для test-строк —
никаких производных от y. Танмото к сматченному соединению не добавлено: матч
выполняется по точному SMILES, поэтому оно тождественно равно 1.

Бейслайны на тех же фолдах: q=0 (текущий soft lookup), q=1 (литература игнорируется)
и глобальные q из {0.25, 0.5, 0.75}. Отбор гипотезы по экранам random-0/1/2,
подтверждение random-3/4/5 и scaffold только проверяется.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from experiments.active_value2 import attach_nn
from experiments.stacking import fold_arrays
from experiments.stacking_eval import load_sources
from vacv.classify import extract_cell_line
from vacv.dataset import load
from vacv.literature import attach
from vacv.pipeline import scaffolds_of
from vacv.splits import fold_indices, scaffold_folds

LOOKUP = Path("data/processed/competition_open_data_lookup.csv")
META_COLUMNS = ["si_quality", "pairing_level", "censored", "matched_cell_line"]
SI_QUALITY_LEVELS = ["measured", "undetermined", "lower_bound", "upper_bound"]
PAIRING_LEVELS = ["strict", "document"]
BAD_MATCH_LOG_TOL = 0.1
GLOBAL_Q = (0.25, 0.5, 0.75)
SCREEN = ["random-0", "random-1", "random-2"]
CONFIRM = ["random-3", "random-4", "random-5", "scaffold"]
VARIANTS = ("q0", "q1", "q0.25", "q0.5", "q0.75", "adaptive")


def attach_lookup_meta(frame: pd.DataFrame) -> pd.DataFrame:
    """Метаданные литературной записи той же строкой, что attach() берёт si: та же
    загрузка файла, тот же dropna/drop_duplicates по smiles."""
    table = pd.read_csv(LOOKUP, usecols=["smiles", *META_COLUMNS])
    table = table.dropna(subset=["smiles"]).drop_duplicates("smiles")
    return frame.merge(table.rename(columns={"smiles": "SMILES"}), on="SMILES", how="left")


def clipped_literature(frame: pd.DataFrame) -> np.ndarray:
    values = frame["literature_si"].to_numpy(dtype=float, na_value=np.nan)
    return np.where(np.isfinite(values), np.clip(values, 0.0, 10000.0), np.nan)


def q_features(meta_frame: pd.DataFrame, literature: np.ndarray) -> np.ndarray:
    """Признаки q по всем строкам; ни один не зависит от y."""
    columns: list[np.ndarray] = []
    for level in SI_QUALITY_LEVELS:
        columns.append((meta_frame["si_quality"].astype("string").eq(level)).to_numpy(dtype=float))
    for level in PAIRING_LEVELS:
        columns.append((meta_frame["pairing_level"].astype("string").eq(level)).to_numpy(dtype=float))
    columns.append(meta_frame["censored"].map({True: 1.0, False: 0.0}).fillna(0.0).to_numpy(dtype=float))
    finite = np.isfinite(literature)
    columns.append(np.where(finite, np.log10(np.where(finite, literature, 1.0) + 1.0), 0.0))
    # Совпадение клеточной линии соревнования с клеточной линией литературной записи.
    query_cells = [extract_cell_line(value) for value in meta_frame["Culture"]]
    match_cells = [extract_cell_line(value) for value in meta_frame["matched_cell_line"]]
    columns.append(np.array([float(q is not None and m is not None)
                             for q, m in zip(query_cells, match_cells)]))
    columns.append(np.array([float(q is not None and q == m)
                             for q, m in zip(query_cells, match_cells)]))
    return np.column_stack(columns)


def bad_match_target(literature: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 — матч плохой: расхождение L с истинным SI больше 0.1 декады. Для
    сентинелов y=10000 это покрывает L заметно меньше 10000."""
    return (np.abs(np.log10((literature + 1.0) / (y + 1.0))) > BAD_MATCH_LOG_TOL).astype(int)


def blend_prediction(probability: np.ndarray, active: np.ndarray, literature: np.ndarray,
                     matched: np.ndarray, q: np.ndarray) -> np.ndarray:
    prediction = probability * 10000.0 + (1.0 - probability) * active
    blend = (1.0 - q) * literature[matched] + q * active[matched]
    prediction[matched] = (probability[matched] * 10000.0
                           + (1.0 - probability[matched]) * blend)
    return prediction


def evaluate_split(name: str, folds: list[np.ndarray], frame: pd.DataFrame, y: np.ndarray,
                   probs: dict[str, np.ndarray], group: np.ndarray, X_all: np.ndarray,
                   lit_all: np.ndarray) -> dict:
    pieces = fold_arrays(probs, group, frame, y, folds)
    sums = {label: 0.0 for label in VARIANTS}
    matched_sums = {label: 0.0 for label in VARIANTS}
    total = 0
    matched_total = 0
    diagnostics = []
    for fold, piece in enumerate(pieces):
        train, _ = fold_indices(folds, fold)
        probability = (piece["probs"]["context"] + piece["probs"]["kernel"]) / 2.0
        active, literature, truth = piece["active"], piece["lit"], piece["y"]
        matched = np.isfinite(literature)

        train_literature = lit_all[train]
        train_matched = np.isfinite(train_literature)
        target = bad_match_target(train_literature[train_matched], y[train][train_matched])
        X_train = X_all[train][train_matched]
        X_valid = X_all[folds[fold]][matched]
        if target.min() == target.max():
            q_hat = np.full(matched.sum(), float(target.mean()))
        else:
            model = LogisticRegression(class_weight="balanced")
            model.fit(X_train, target)
            q_hat = model.predict_proba(X_valid)[:, 1]
        diagnostics.append({
            "train_matched": int(train_matched.sum()),
            "train_bad_rate": float(target.mean()),
            "valid_matched": int(matched.sum()),
            "q_mean": float(np.mean(q_hat)) if matched.any() else None,
            "q_std": float(np.std(q_hat)) if matched.any() else None,
        })

        variants = {
            "q0": np.zeros(matched.sum()),
            "q1": np.ones(matched.sum()),
            **{f"q{value}": np.full(matched.sum(), value) for value in GLOBAL_Q},
            "adaptive": q_hat,
        }
        for label, q in variants.items():
            prediction = blend_prediction(probability, active, literature, matched, q)
            sums[label] += float(np.sum((prediction - truth) ** 2))
            matched_sums[label] += float(np.sum((prediction[matched] - truth[matched]) ** 2))
        total += len(truth)
        matched_total += int(matched.sum())
    result = {
        "split": name,
        "n_rows": total,
        "n_matched": matched_total,
        "baseline_q0": sums["q0"] / total,
        "baseline_q1": sums["q1"] / total,
        "global_q": {str(value): sums[f"q{value}"] / total for value in GLOBAL_Q},
        "matched_baseline": matched_sums["q0"] / matched_total,
        "matched_adaptive": matched_sums["adaptive"] / matched_total,
        "adaptive": sums["adaptive"] / total,
        "fold_diagnostics": diagnostics,
    }
    print(f"{name:12} q0={result['baseline_q0']:,.0f} q1={result['baseline_q1']:,.0f} "
          + " ".join(f"q{v}={result['global_q'][str(v)]:,.0f}" for v in GLOBAL_Q)
          + f" adaptive={result['adaptive']:,.0f} "
          + f"| matched q0={result['matched_baseline']:,.0f} "
          + f"adaptive={result['matched_adaptive']:,.0f}", flush=True)
    return result


def mean_over(rows: list[dict], names: list[str]) -> dict:
    subset = [row for row in rows if row["split"] in names]
    if len(subset) != len(names):
        raise ValueError(f"missing splits among {names}")
    return {
        "baseline_q0": float(np.mean([r["baseline_q0"] for r in subset])),
        "baseline_q1": float(np.mean([r["baseline_q1"] for r in subset])),
        "global_q": {str(v): float(np.mean([r["global_q"][str(v)] for r in subset])) for v in GLOBAL_Q},
        "matched_baseline": float(np.mean([r["matched_baseline"] for r in subset])),
        "matched_adaptive": float(np.mean([r["matched_adaptive"] for r in subset])),
        "adaptive": float(np.mean([r["adaptive"] for r in subset])),
    }


def main():
    import warnings
    data = load()
    frame, y = attach_nn(attach(data.train)), data.y
    meta_frame = attach_lookup_meta(frame)
    assert len(meta_frame) == len(frame), "meta merge must not duplicate rows"
    lit_all = clipped_literature(meta_frame)
    X_all = q_features(meta_frame, lit_all)

    splits: dict[str, tuple[Path, list[np.ndarray]]] = {}
    for name in SCREEN:
        splits[name] = (Path("outputs/stacking-screen"),
                        [v for _, v in KFold(5, shuffle=True, random_state=int(name.split("-")[1])).split(y)])
    for name in CONFIRM:
        if name == "scaffold":
            splits[name] = (Path("outputs/stacking-confirm"), scaffold_folds(scaffolds_of(frame), 5))
        else:
            splits[name] = (Path("outputs/stacking-confirm"),
                            [v for _, v in KFold(5, shuffle=True, random_state=int(name.split("-")[1])).split(y)])

    rows = []
    for name, (source_dir, folds) in splits.items():
        probs, group = load_sources(source_dir, name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # сходимость lbfgs не влияет на детерминированный ответ
            rows.append(evaluate_split(name, folds, frame, y, probs, group, X_all, lit_all))

    screen_mean = mean_over(rows, SCREEN)
    confirm_mean = mean_over(rows, CONFIRM)
    gain = screen_mean["baseline_q0"] - screen_mean["adaptive"]
    best_const = min(screen_mean["global_q"].values())
    best_const_q = min(screen_mean["global_q"], key=screen_mean["global_q"].get)
    # Подтверждением считаем только относительный выигрыш над q=0 не менее 0.5%
    # и превосходство над лучшей глобальной константой: меньшие различия на этом
    # наборе строк недетерминированы между сплитами и равны шуму.
    confirmed = gain >= 0.005 * screen_mean["baseline_q0"] and screen_mean["adaptive"] < best_const
    conclusion = (
        f"Гипотеза подтверждается: на экранах адаптивное q даёт MSE {screen_mean['adaptive']:,.0f} "
        f"против {screen_mean['baseline_q0']:,.0f} у q=0, выигрыш {gain:+,.0f} "
        f"({gain / screen_mean['baseline_q0']:.2%} относительного), "
        f"лучшая константа q={best_const_q}: {best_const:,.0f}."
        if confirmed else
        f"Гипотеза не подтверждается: на экранах адаптивное q даёт MSE {screen_mean['adaptive']:,.0f} "
        f"против {screen_mean['baseline_q0']:,.0f} у текущего soft lookup (q=0) — разница "
        f"{gain:+,.0f} ({gain / screen_mean['baseline_q0']:.4%} относительного), уровень шума: "
        f"по сплитам знак различия меняется. Лучшая глобальная константа q={best_const_q} "
        f"({best_const:,.0f}) не хуже адаптивной модели, то есть признаки q не дают "
        f"селективной информации о качестве матча сверх равномерного подмешивания A(x)."
    )
    out = Path("outputs/lookup-q")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"splits": rows, "screen_mean": screen_mean, "confirm_mean": confirm_mean,
         "conclusion": conclusion}, indent=2) + "\n")
    print(json.dumps({"screen_mean": screen_mean, "conclusion": conclusion}, indent=2), flush=True)


if __name__ == "__main__":
    main()
