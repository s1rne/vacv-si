"""Модели на химических признаках: классификатор неактивности и регрессор SI активных.

Классификатор решает метрику: промах на неактивном соединении стоит порядка 10⁸, ошибка
в разы на активном — порядка 10². Регрессор проверяется отдельно, чтобы убедиться, что
константа по обучающей части фолда не хуже обученного прогноза.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

from experiments.chem_common import DATA, FOLDS, LABELS, Y, classifier_oof, matrix
from experiments.chem_zoo import CLASSIFIERS, FINGERPRINT_ONLY, REGRESSORS
from vacv.chemistry import RECOMMENDED
from vacv.evaluate import competition_mse, out_of_fold, splits
from vacv.splits import fold_indices

FINGERPRINTS = ("ecfp2", "avalon", "maccs")


def ceiling() -> None:
    """Потолок метрики: идеальный классификатор заглушек и константа для активных.

    Нужен как масштаб. Метрика квадратичная, и всё, что можно выиграть, лежит в
    классификации заглушек, а не в прогнозе SI активных.
    """
    print("— потолок и пол —")
    print(f"{'конфигурация':34} {'MSE rn':>12} {'MSE sc':>12}")
    for label, probability in (("идеальный классификатор", LABELS.astype(float)),
                               ("базовая доля заглушек", np.full(len(Y), LABELS.mean()))):
        scores = [competition_mse(probability, Y, FOLDS[split]) for split in ("random", "scaffold")]
        print(f"{label:34} {scores[0]:12,.0f} {scores[1]:12,.0f}")


def classifiers() -> None:
    print("— классификатор неактивности, признаки " + "+".join(RECOMMENDED) + " —")
    print(f"{'модель':14} {'AUC rn':>7} {'MSE rn':>12} {'AUC sc':>7} {'MSE sc':>12}")
    full, fingerprint = matrix(RECOMMENDED), matrix(FINGERPRINTS)
    for name in CLASSIFIERS:
        x = fingerprint if name in FINGERPRINT_ONLY else full
        scores = []
        for split in ("random", "scaffold"):
            probability = classifier_oof(name, x, FOLDS[split])
            scores += [roc_auc_score(LABELS, probability), competition_mse(probability, Y, FOLDS[split])]
        mark = " (отпечатки)" if name in FINGERPRINT_ONLY else ""
        print(f"{name + mark:14} {scores[0]:7.3f} {scores[1]:12,.0f} {scores[2]:7.3f} {scores[3]:12,.0f}")


def active_oof(name: str, x: np.ndarray, folds: list[np.ndarray], log: bool) -> np.ndarray:
    """Прогноз SI для активных: модель учится только на активных строках обучающей части."""
    prediction = np.zeros(len(Y))
    target = np.log10(np.clip(Y, 1e-3, None)) if log else Y
    for fold in range(len(folds)):
        train_index, validation_index = fold_indices(folds, fold)
        active = train_index[LABELS[train_index] == 0]
        model = REGRESSORS[name]().fit(x[active], target[active])
        value = model.predict(x[validation_index])
        prediction[validation_index] = np.power(10.0, np.clip(value, -3, 4.3)) if log else value
    return np.clip(prediction, 0.0, 10_000.0)


def regressors(probability: dict[str, np.ndarray]) -> None:
    print("\n— оценка SI активных при фиксированном классификаторе —")
    print(f"{'оценка':22} {'MSE rn':>12} {'MSE sc':>12}")
    full, fingerprint = matrix(RECOMMENDED), matrix(FINGERPRINTS)
    rows = [("константа фолда", None, False)]
    rows += [(f"{name}, log10 SI", name, True) for name in REGRESSORS]
    rows += [("lgbm, исходная шкала", "lgbm", False)]
    for label, name, log in rows:
        scores = []
        for split in ("random", "scaffold"):
            if name is None:
                active = None
            else:
                x = fingerprint if name in ("svr_tanimoto", "knn_tanimoto") else full
                active = active_oof(name, x, FOLDS[split], log)
            scores.append(competition_mse(probability[split], Y, FOLDS[split], active))
        print(f"{label:22} {scores[0]:12,.0f} {scores[1]:12,.0f}")


def stability(seeds: int = 5) -> None:
    """Разброс от разбиения и сида: без него разница в третьем знаке AUC ничего не значит."""
    print("\n— устойчивость отобранной конфигурации по пяти разбиениям —")
    x = matrix(RECOMMENDED)
    for split in ("random", "scaffold"):
        auc, mse = [], []
        for seed in range(seeds):
            folds = splits(DATA.train, Y, seed=seed)[split]
            probability = classifier_oof("extra", x, folds)
            auc.append(roc_auc_score(LABELS, probability))
            mse.append(competition_mse(probability, Y, folds))
        print(f"{split:10} AUC {np.mean(auc):.3f} ± {np.std(auc):.3f}   "
              f"MSE {np.mean(mse):,.0f} ± {np.std(mse):,.0f}")


def bagged() -> None:
    """Усреднение ExtraTrees по пяти сидам и смесь с декоррелированными моделями."""
    print("\n— как сильнее выжать одну химию —")
    print(f"{'вариант':34} {'AUC rn':>7} {'MSE rn':>12} {'AUC sc':>7} {'MSE sc':>12}")
    x = matrix(RECOMMENDED)
    variants = {}
    for split in ("random", "scaffold"):
        folds = FOLDS[split]
        seeds = np.mean([seed_oof(x, folds, seed) for seed in range(5)], axis=0)
        variants.setdefault("extra, один сид", {})[split] = classifier_oof("extra", x, folds)
        variants.setdefault("extra, пять сидов", {})[split] = seeds
        for partner in ("logreg", "rf", "knn_tanimoto"):
            other = classifier_oof(partner, matrix(FINGERPRINTS) if partner == "knn_tanimoto" else x, folds)
            variants.setdefault(f"пять сидов + {partner}", {})[split] = 0.5 * seeds + 0.5 * other
    for label, probability in variants.items():
        scores = []
        for split in ("random", "scaffold"):
            scores += [roc_auc_score(LABELS, probability[split]),
                       competition_mse(probability[split], Y, FOLDS[split])]
        print(f"{label:34} {scores[0]:7.3f} {scores[1]:12,.0f} {scores[2]:7.3f} {scores[3]:12,.0f}")


def seed_oof(x: np.ndarray, folds: list[np.ndarray], seed: int) -> np.ndarray:
    def predict(train_index: np.ndarray, validation_index: np.ndarray) -> np.ndarray:
        model = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=seed)
        model.fit(x[train_index], LABELS[train_index])
        return model.predict_proba(x[validation_index])[:, 1]

    return out_of_fold(predict, LABELS, folds)


def main() -> None:
    ceiling()
    classifiers()
    bagged()
    x = matrix(RECOMMENDED)
    probability = {split: classifier_oof("extra", x, FOLDS[split]) for split in FOLDS}
    regressors(probability)
    stability()


if __name__ == "__main__":
    main()
