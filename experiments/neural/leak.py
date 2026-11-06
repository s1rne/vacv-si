"""Откуда берётся выигрыш языковых моделей: из химии или из манеры записи SMILES.

Файл соревнования собран из статей, и строки SMILES в нём записаны так, как их записали
авторы: 141 строка из 320 не совпадает с канонической записью RDKit. Внутри одной статьи
серия соединений обычно либо активна, либо нет, поэтому манера записи косвенно указывает
на метку. Модель, читающая строку (а не граф), может опознавать именно её.

Проверка не требует нейросети. Если тот же выигрыш даёт логистическая регрессия на
символьных n-граммах строки и исчезает после канонизации, дело в записи, а не
в представлении молекулы. Крайний случай — один бит «строка уже канонична».

Признаки присутствия в открытых базах тут не используются: вопрос только про строку.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from vacv.neural import store
from vacv.neural.augment import randomized
from vacv.protocol import HEADER, Result, fold_auc, score, task

RDLogger.DisableLog("rdApp.*")


def ngram_probe(job, split: str, texts: list[str], name: str) -> Result:
    import time

    probability = np.zeros(len(job.y))
    started = time.perf_counter()
    for train_index, validation_index in job.splits(split):
        model = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            LogisticRegression(C=1.0, max_iter=5000, random_state=0))
        model.fit([texts[i] for i in train_index], job.labels[train_index])
        probability[validation_index] = model.predict_proba([texts[i] for i in validation_index])[:, 1]
    auc, mse = score(job.labels, job.y, probability, job.active_mean)
    return Result(name, split, auc, mse, time.perf_counter() - started, probability)


def main() -> None:
    job = task()
    smiles = list(job.frame["SMILES"])
    canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(text)) for text in smiles]
    shuffled = [randomized(text, 2, seed=index)[-1] for index, text in enumerate(smiles)]

    flag = np.array([a == b for a, b in zip(smiles, canonical)], dtype=float)
    print(f"один бит «строка уже канонична»: AUC {roc_auc_score(job.labels, 1 - flag):.3f}")
    print(f"  неканоничных строк {int((1 - flag).sum())}, среди них неактивных "
          f"{int(job.labels[flag == 0].sum())} ({job.labels[flag == 0].mean():.1%})")
    print(f"  каноничных строк {int(flag.sum())}, среди них неактивных "
          f"{int(job.labels[flag == 1].sum())} ({job.labels[flag == 1].mean():.1%})\n")

    variants = {"как в файле": smiles, "канонический RDKit": canonical, "случайная запись": shuffled}
    print(HEADER + f" {'AUC фолд':>9}")
    results = []
    for name, texts in variants.items():
        for split in ("random", "scaffold"):
            result = ngram_probe(job, split, texts, f"char n-gram tf-idf [{name}]")
            results.append(result)
            mean, _ = fold_auc(job, split, result.probability)
            print(result.row() + f" {mean:9.3f}")
    store.save("leak", results)


if __name__ == "__main__":
    main()
