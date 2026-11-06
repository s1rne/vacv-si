"""Проверки, без которых выигрыш замороженных эмбеддингов нельзя принимать.

Линейный зонд поверх ChemBERTa и MoLFormer обходит рабочий ансамбль на отпечатках
с большим отрывом, и этого не должно быть по всей собранной литературе. Значит, сначала
нужно исключить три объяснения, не связанные с качеством представления.

1. **Голова, а не представление.** Отсчёт — ExtraTrees, а зонд — логистическая регрессия
   со стандартизацией. Поэтому та же логистическая регрессия ставится на отпечатки
   Моргана, MACCS и дескрипторы RDKit.
2. **Стиль записи SMILES.** 141 строка из 320 не канонична. Соединения одной статьи
   попали в файл так, как их записали авторы, а внутри статьи серия обычно либо активна,
   либо нет. Языковая модель читает строку, а не молекулу, и может опознавать манеру
   записи. Контроль: те же молекулы, переписанные каноническим SMILES RDKit, и те же
   молекулы в случайной записи.
3. **Завышение общего AUC.** Доля неактивных по фолдам от 0.17 до 0.38, поэтому общий AUC
   по склеенным out-of-fold вероятностям печатается вместе со средним по фолдам.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger

from vacv.baseline import baselines
from vacv.features import descriptors, maccs, morgan_counts, parse
from vacv.neural import embed, store
from vacv.neural.augment import randomized
from vacv.neural.probe import probe
from vacv.protocol import fold_auc, task

RDLogger.DisableLog("rdApp.*")

HEADER = (f"{'представление':44} {'разбиение':9} {'AUC':>6} {'AUC фолд':>9} {'±':>5} "
          f"{'MSE':>13} {'сек':>7}")

WRITINGS = {
    "как в файле": lambda smiles: list(smiles),
    "канонический RDKit": lambda smiles: [Chem.MolToSmiles(Chem.MolFromSmiles(text)) for text in smiles],
    "случайная запись": lambda smiles: [randomized(text, 2, seed=index)[-1]
                                        for index, text in enumerate(smiles)],
}
LANGUAGE_MODELS = ("chemberta-77m-mlm", "molformer-xl")


def classic(frame) -> dict[str, np.ndarray]:
    molecules = [parse(text) for text in frame["SMILES"]]
    return {
        "morgan counts": np.vstack([morgan_counts(mol) for mol in molecules]),
        "MACCS": np.vstack([maccs(mol) for mol in molecules]),
        "дескрипторы RDKit": np.vstack([descriptors(mol) for mol in molecules]),
        "morgan + MACCS": np.vstack([np.concatenate([morgan_counts(mol), maccs(mol)])
                                     for mol in molecules]),
    }


def report(job, split, result) -> None:
    mean, deviation = fold_auc(job, split, result.probability)
    print(f"{result.model:44} {split:9} {result.auc:6.3f} {mean:9.3f} {deviation:5.3f} "
          f"{result.mse:13,.0f} {result.seconds:7.1f}", flush=True)


def main() -> None:
    job = task()
    smiles = list(job.frame["SMILES"])
    blocks = classic(job.frame)
    results = []

    print(HEADER)
    for split in ("random", "scaffold"):
        for result in baselines(job, split):
            report(job, split, result)
        for name, matrix in blocks.items():
            result = probe(job, split, matrix, name, "linear")
            results.append(result)
            report(job, split, result)

    for writing, rewrite in WRITINGS.items():
        texts = rewrite(smiles)
        changed = sum(a != b for a, b in zip(smiles, texts))
        print(f"\n# запись «{writing}»: отличается от файла в {changed} строках из {len(texts)}")
        for name in LANGUAGE_MODELS:
            embedding = embed.load(name, texts)
            matrices = {f"{name} [{writing}]": embedding.mean}
            if writing == "канонический RDKit":
                matrices[f"{name} + morgan [{writing}]"] = np.hstack(
                    [embedding.mean, blocks["morgan counts"]])
            for label, matrix in matrices.items():
                for split in ("random", "scaffold"):
                    result = probe(job, split, matrix, label, "linear",
                                   extra_seconds=embedding.seconds)
                    results.append(result)
                    report(job, split, result)

    store.save("controls", results)


if __name__ == "__main__":
    main()
