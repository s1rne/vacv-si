"""Качество голов IC50 и CC50 на внешнем корпусе.

Три разбиения: случайное, по остовам Мурко и по публикации (DOI). Последнее отвечает
на вопрос, который важнее остальных для переноса: работает ли модель на статье, которой
она не видела, то есть на другом протоколе и другой лаборатории.

Метрики считаются только по точным значениям контрольной части: иначе цензурированный
и обычный режимы сравнивались бы на разных метках. Дополнительно печатается односторонняя
ошибка по цензурированным строкам: там штраф начисляется только за прогноз ниже границы.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from vacv.endpoints import load_corpus
from vacv.features import parse, scaffold
from vacv.heads import ENDPOINTS, EndpointHeads
from vacv.splits import fold_indices, scaffold_folds


def document_folds(documents: pd.Series, n_splits: int = 5) -> list[np.ndarray]:
    """Фолды, не делящие публикацию: внутри статьи протокол один и серия однородна."""
    return scaffold_folds(documents.fillna("_none").astype(str).tolist(), n_splits=n_splits)


def splits(corpus: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    scaffolds = [scaffold(parse(smiles)) for smiles in corpus["smiles"]]
    return {
        "случайное": [index for _, index in KFold(5, shuffle=True, random_state=0).split(corpus)],
        "остовы": scaffold_folds(scaffolds, n_splits=5),
        "публикации": document_folds(corpus["document_id"]),
    }


def out_of_fold(corpus: pd.DataFrame, folds: list[np.ndarray], censored: bool) -> pd.DataFrame:
    prediction = pd.DataFrame(index=corpus.index, columns=["log_ic50", "log_cc50"], dtype=float)
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        heads = EndpointHeads(censored=censored).fit(corpus.iloc[train_index])
        predicted = heads.predict_corpus(corpus.iloc[validation_index])
        prediction.iloc[validation_index] = predicted[["log_ic50", "log_cc50"]].to_numpy()
    return prediction


def score(corpus: pd.DataFrame, prediction: pd.DataFrame, endpoint: str) -> tuple[float, float, float]:
    truth = corpus[f"log_{endpoint}"].to_numpy(float)
    predicted = prediction[f"log_{endpoint}"].to_numpy(float)
    exact = ~corpus[f"{endpoint}_censored"].to_numpy(bool)
    rmse = float(np.sqrt(np.mean((predicted[exact] - truth[exact]) ** 2)))
    rho = float(spearmanr(predicted[exact], truth[exact]).statistic)
    shortfall = np.clip(truth[~exact] - predicted[~exact], 0, None)
    one_sided = float(np.sqrt(np.mean(shortfall**2))) if shortfall.size else float("nan")
    return rmse, rho, one_sided


def main() -> None:
    corpus = load_corpus()
    print(f"корпус {len(corpus)} соединений; цензурировано IC50 "
          f"{corpus['ic50_censored'].mean():.0%}, CC50 {corpus['cc50_censored'].mean():.0%}")
    header = f"{'разбиение':12} {'режим':14} {'эндпоинт':9} {'RMSE':>6} {'Spearman':>9} {'одностор.':>10}"
    print(header)
    for name, folds in splits(corpus).items():
        for censored, label in ((True, "цензура AFT"), (False, "только точные")):
            prediction = out_of_fold(corpus, folds, censored)
            for endpoint in ENDPOINTS:
                rmse, rho, one_sided = score(corpus, prediction, endpoint)
                print(f"{name:12} {label:14} {endpoint:9} {rmse:6.3f} {rho:9.3f} {one_sided:10.3f}")
        print()


if __name__ == "__main__":
    main()
