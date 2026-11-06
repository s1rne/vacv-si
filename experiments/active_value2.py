"""По-строчная оценка среднего SI активных вместо глобальной константы.

Декомпозиция MSE текущего чекпоинта показывает: единственная строка с настоящим
SI = 36066.667 даёт около 4M из 8.6M MSE, потому что условное среднее активных —
глобальная константа 285.7. Между тем у 22.train строк во внешних базах известен IC50,
и по закону SI = CC50 / IC50 такое значение предсказуемо: медиана неявного CC50
(SI × IC50) по этим строкам около 200, то есть для IC50 = 0.03 ожидаемый SI ~ 10^4.

Здесь A(x) — оценка E[SI | соединение активно] по признакам присутствия во внешних
базах, групповым средним и ближайшим соседям. Она обучается только на активных строках
outer-train каждого фолда. Финальный прогноз: p × 10000 + (1 − p) × A(x), затем
жёсткая или условная литературная подстановка — как в основном пайплайне.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from vacv.dataset import load
from vacv.inference import predict_ensemble
from vacv.literature import PRESENCE_FEATURES, attach
from vacv.pipeline import INACTIVE
from vacv.splits import fold_indices, scaffold_folds


def _log_column(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log10 значения и флаг его наличия; пропуск -> 0 и флаг 0."""
    finite = np.isfinite(values) & (values > 0)
    log = np.where(finite, np.log10(np.where(finite, values, 1.0)), 0.0)
    return log, finite.astype(float)


def active_features(frame: pd.DataFrame, group_mean: np.ndarray) -> np.ndarray:
    """Признаки для оценки активного SI: внешние измерения, соседи, группа."""
    columns = {}
    ic_log, ic_has = _log_column(frame["best_ic50_um"].to_numpy(dtype=float, na_value=np.nan))
    cc_log, cc_has = _log_column(frame["best_cc50_um"].to_numpy(dtype=float, na_value=np.nan))
    columns["ic50_log"] = ic_log
    columns["ic50_has"] = ic_has
    columns["cc50_log"] = cc_log
    columns["cc50_has"] = cc_has
    for name in ("nn1", "nn2", "nn3"):
        tanimoto = frame[f"{name}_tanimoto"].to_numpy(dtype=float, na_value=np.nan)
        si = frame[f"{name}_si"].to_numpy(dtype=float, na_value=np.nan)
        si_log, si_has = _log_column(si)
        columns[f"{name}_sim"] = np.where(np.isfinite(tanimoto), tanimoto, 0.0)
        columns[f"{name}_si_log"] = si_log
        columns[f"{name}_si_has"] = si_has
    for name in ("n_antiviral", "n_cytotox", "n_records_skeleton"):
        values = frame[name].to_numpy(dtype=float, na_value=np.nan)
        columns[name] = np.log1p(np.where(np.isfinite(values), values, 0.0))
    columns["group_mean_log"] = group_mean
    return np.column_stack([columns[name] for name in sorted(columns)])


def _shrunk_group_log_mean(train_frame: pd.DataFrame, train_si: np.ndarray,
                           frame: pd.DataFrame, smoothing: float = 3.0) -> np.ndarray:
    """Среднее log10 SI активных обучающих строк группы, сглаженное к общему среднему."""
    labels = train_si != INACTIVE
    values = np.log10(np.clip(train_si[labels], 1e-6, None))
    keys = list(map(tuple, train_frame.loc[labels, ["strain", "Culture"]].values))
    total = float(values.mean()) if len(values) else 0.0
    sums: dict[tuple, list[float]] = {}
    for key, value in zip(keys, values):
        sums.setdefault(key, []).append(value)
    encoded = {key: (len(v) * float(np.mean(v)) + smoothing * total) / (len(v) + smoothing)
               for key, v in sums.items()}
    query = list(map(tuple, frame[["strain", "Culture"]].values))
    return np.array([encoded.get(key, total) for key in query])


class ActiveValueModel:
    """Ridge по log10 SI активных строк outer-train; сырое значение — с калибровкой среднего."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, train_frame: pd.DataFrame, train_si: np.ndarray) -> "ActiveValueModel":
        labels = train_si != INACTIVE
        si = train_si[labels]
        self.train_frame_ = train_frame
        self.train_si_ = train_si
        group_mean = _shrunk_group_log_mean(train_frame, train_si, train_frame)
        X = active_features(train_frame, group_mean)[labels]
        y = np.log10(np.clip(si, 1e-6, None))
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(X), y)
        fitted = self.model_.predict(self.scaler_.transform(X))
        # Калибровка: среднее сырое предсказание на обучении равно среднему SI активных.
        self.scale_ = float(si.mean() / max(np.mean(10.0 ** fitted), 1e-12))
        self.fallback_ = float(si.mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        group_mean = _shrunk_group_log_mean(self.train_frame_, self.train_si_, frame)
        X = active_features(frame, group_mean)
        log = self.model_.predict(self.scaler_.transform(X))
        value = 10.0 ** log * self.scale_
        return np.where(np.isfinite(value) & (value > 0), value, self.fallback_)


def active_value_constant(train_frame: pd.DataFrame, train_si: np.ndarray) -> ActiveValueModel:
    """Текущее поведение: глобальное среднее активных."""
    model = ActiveValueModel.__new__(ActiveValueModel)
    labels = train_si != INACTIVE
    model.fallback_ = float(train_si[labels].mean())
    model.scale_ = 1.0
    model.train_frame_ = train_frame
    model.train_si_ = train_si

    def predict(frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), model.fallback_)

    model.predict = predict  # type: ignore[method-assign]
    return model


def group_mean_model(smoothing: float = 3.0):
    """Только сглаженное среднее активных по группе."""

    class GroupModel:
        def fit(self, train_frame: pd.DataFrame, train_si: np.ndarray):
            self.train_frame_ = train_frame
            self.train_si_ = train_si
            return self

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            labels = self.train_si_ != INACTIVE
            si = self.train_si_[labels]
            keys = list(map(tuple, self.train_frame_.loc[labels, ["strain", "Culture"]].values))
            total = float(si.mean())
            sums: dict[tuple, list[float]] = {}
            for key, value in zip(keys, si):
                sums.setdefault(key, []).append(value)
            encoded = {key: (len(v) * float(np.mean(v)) + smoothing * total) / (len(v) + smoothing)
                       for key, v in sums.items()}
            query = list(map(tuple, frame[["strain", "Culture"]].values))
            return np.array([encoded.get(key, total) for key in query])

    return GroupModel()


class PhysicalActiveValue:
    """A(x) из закона SI = CC50 / IC50, а не из регрессии.

    Если во внешних базах известен IC50 соединения, то SI = CC50 / IC50, и вся
    неопределённость сидит в цитотоксичности. CC50_hat — взвешенное по сходству
    среднее измеренных CC50 похожих соединений внешней базы (тысячи измерений),
    стянутое к среднему по той же клеточной линии. Если известен только CC50,
    нужна обратная величина: A = CC50 × E[1/IC50] похожих соединений. Если не
    известно ничего — сглаженное среднее активных по группе.

    Сосед, совпадающий с самой строкой по InChIKey14, исключается: его CC50 —
    это то, что мы и оцениваем, а не признак.
    """

    def __init__(self, k: int = 10, min_sim: float = 0.05, power: float = 2.0,
                 shrink: float = 2.0, train_power: float = 1.0, both_rule: str = "ratio"):
        self.k = k
        self.min_sim = min_sim
        self.power = power
        self.shrink = shrink
        self.train_power = train_power
        self.both_rule = both_rule

    # --- подготовка внешней базы -------------------------------------------------
    @staticmethod
    def _fingerprints(smiles_list):
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        rows = []
        for text in smiles_list:
            mol = Chem.MolFromSmiles(text)
            rows.append(gen.GetFingerprint(mol) if mol is not None else None)
        return rows

    @staticmethod
    def _tanimoto_matrix(query_fps, db_fps):
        from rdkit import DataStructs
        matrix = np.zeros((len(query_fps), len(db_fps)))
        for i, left in enumerate(query_fps):
            if left is None:
                continue
            for j, right in enumerate(db_fps):
                if right is not None:
                    matrix[i, j] = DataStructs.TanimotoSimilarity(left, right)
        return matrix

    def fit(self, train_frame: pd.DataFrame, train_si: np.ndarray) -> "PhysicalActiveValue":
        from vacv.classify import extract_cell_line
        labels = train_si != INACTIVE
        self.train_frame_ = train_frame.reset_index(drop=True)
        self.train_si_ = train_si
        train_active = self.train_frame_.loc[labels].copy()
        train_active["si"] = train_si[labels]

        ic50 = pd.to_numeric(train_active["best_ic50_um"], errors="coerce").to_numpy()
        cc50 = pd.to_numeric(train_active["best_cc50_um"], errors="coerce").to_numpy()
        ic_known = np.isfinite(ic50) & (ic50 > 0)
        cc_known = np.isfinite(cc50) & (cc50 > 0)
        # Неявный CC50 известен там, где известен IC50: CC50 = SI × IC50.
        # Массивы по всем обучающим строкам (включая сентинелы как NaN).
        ic50_all = pd.to_numeric(self.train_frame_["best_ic50_um"], errors="coerce").to_numpy()
        cc50_all = pd.to_numeric(self.train_frame_["best_cc50_um"], errors="coerce").to_numpy()
        si_all = np.where(train_si != INACTIVE, train_si, np.nan)
        self.train_cc50_values_ = si_all * ic50_all
        inv_full = np.full(len(self.train_frame_), np.nan)
        known_all = np.isfinite(ic50_all) & (ic50_all > 0) & (train_si != INACTIVE)
        inv_full[known_all] = 1.0 / ic50_all[known_all]
        self.train_inv_values_ = inv_full
        # Отношение SI / CC50 для строк с известным CC50: обучающая цель для 1/IC50.
        ratio_full = np.full(len(self.train_frame_), np.nan)
        ratio_known = np.isfinite(cc50_all) & (cc50_all > 0) & (train_si != INACTIVE)
        ratio_full[ratio_known] = train_si[ratio_known] / cc50_all[ratio_known]
        self.train_ratio_values_ = ratio_full

        # Внешняя база: измеренные CC50 и IC50.
        db = pd.read_csv("data/processed/vacv_si.csv")
        db_cc = db[db["cc50_relation"].eq("=") & db["cc50"].notna()].drop_duplicates("inchikey14")
        db_ic = db[db["ic50_relation"].eq("=") & db["ic50"].notna()].drop_duplicates("inchikey14")
        self.db_cc50_ = db_cc["cc50"].to_numpy(dtype=float)
        self.db_cc_cell_ = db_cc["cell_line"].astype(str).tolist()
        self.db_inv_ic50_ = 1.0 / db_ic["ic50"].to_numpy(dtype=float)
        self.db_ic_cell_ = db_ic["cell_line"].astype(str).tolist()

        # Приор по клеточной линии запроса: среднее CC50 той же линии, иначе глобальное.
        global_cc = float(np.mean(self.db_cc50_))
        self.cc_prior_ = global_cc
        self.cc_by_cell_ = {}
        grouped = pd.DataFrame({"cell": self.db_cc_cell_, "cc": self.db_cc50_}).groupby("cell")["cc"]
        for cell, values in grouped:
            if len(values) >= 5:
                self.cc_by_cell_[cell] = float(values.mean())

        # Отпечатки: база + train (для исключения самих строк) + любые запросы при predict.
        self.db_fp_cc_ = self._fingerprints(db_cc["smiles"].tolist())
        self.db_fp_ic_ = self._fingerprints(db_ic["smiles"].tolist())
        self.db_cc_keys_ = db_cc["inchikey14"].tolist()
        self.db_ic_keys_ = db_ic["inchikey14"].tolist()
        self.train_fp_ = self._fingerprints(self.train_frame_["SMILES"].tolist())
        self.train_keys_ = self.train_frame_["inchikey14"].tolist()
        self._extract_cell_line = extract_cell_line
        return self

    def _cell_prior(self, cell: str | None) -> float:
        if cell and cell in self.cc_by_cell_:
            return self.cc_by_cell_[cell]
        return self.cc_prior_

    def _estimate(self, frame: pd.DataFrame, db_fp, db_values, db_keys,
                  train_values, prior: np.ndarray | float) -> np.ndarray:
        """Взвешенное среднее значений похожих соединений: база + train, стянутое к prior."""
        query_fp = self._fingerprints(frame["SMILES"].tolist())
        query_keys = frame["inchikey14"].tolist()
        sim = self._tanimoto_matrix(query_fp, db_fp)
        train_sim = self._tanimoto_matrix(query_fp, self.train_fp_)
        if np.isscalar(prior):
            prior = np.full(len(frame), float(prior))
        known = np.isfinite(train_values)
        train_prior = float(np.mean(train_values[known])) if known.any() else float(np.mean(prior))
        estimates = np.empty(len(frame))
        for i in range(len(frame)):
            weights = np.where(sim[i] > self.min_sim, sim[i] ** self.power, 0.0)
            if query_keys[i]:
                weights = np.where(np.array([k == query_keys[i] for k in db_keys]), 0.0, weights)
            train_w = np.where(known & (train_sim[i] > self.min_sim), train_sim[i] ** self.train_power, 0.0)
            if query_keys[i]:
                train_w = np.where(np.array([k == query_keys[i] for k in self.train_keys_]), 0.0, train_w)
            total = weights.sum() + train_w.sum() + self.shrink
            if total <= self.shrink:
                estimates[i] = train_prior if known.any() else prior[i]
                continue
            db_part = float(np.dot(weights, db_values)) if weights.any() else 0.0
            train_part = float(np.nansum(train_w * np.nan_to_num(train_values, nan=0.0))) if train_w.any() else 0.0
            estimates[i] = (db_part + train_part + self.shrink * prior[i]) / total
        return estimates

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        ic50 = pd.to_numeric(frame["best_ic50_um"], errors="coerce").to_numpy()
        cc50 = pd.to_numeric(frame["best_cc50_um"], errors="coerce").to_numpy()
        labels = self.train_si_ != INACTIVE
        fallback = float(self.train_si_[labels].mean())
        result = np.full(len(frame), fallback)
        ic_known = np.isfinite(ic50) & (ic50 > 0)
        cc_known = np.isfinite(cc50) & (cc50 > 0)
        both = ic_known & cc_known
        if self.both_rule == "ratio" and both.any():
            result[both] = cc50[both] / ic50[both]
        est = ic_known & ~both
        if est.any():
            prior = np.array([self._cell_prior(self._extract_cell_line(c)) for c in frame.loc[est, "Culture"]])
            cc_hat = self._estimate(frame[est], self.db_fp_cc_, self.db_cc50_, self.db_cc_keys_,
                                    self.train_cc50_values_, prior)
            result[est] = cc_hat / ic50[est]
        only_cc = cc_known & ~ic_known
        if only_cc.any():
            known_ratio = self.train_ratio_values_[np.isfinite(self.train_ratio_values_)]
            ratio_prior = float(np.mean(known_ratio)) if len(known_ratio) else fallback / 100.0
            ratio_hat = self._estimate(frame[only_cc], self.train_fp_, self.train_ratio_values_,
                                       self.train_keys_, self.train_ratio_values_, ratio_prior)
            result[only_cc] = cc50[only_cc] * ratio_hat
        result = np.where(np.isfinite(result) & (result > 0), result, fallback)
        return np.clip(result, 0.0, None)


def hard_lookup(prediction: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    literature = frame["literature_si"].to_numpy(dtype=float, na_value=np.nan)
    found = np.isfinite(literature)
    result = prediction.copy()
    result[found] = np.clip(literature[found], 0.0, 10000.0)
    return result


def soft_lookup(prediction: np.ndarray, probability: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    literature = frame["literature_si"].to_numpy(dtype=float, na_value=np.nan)
    found = np.isfinite(literature)
    result = prediction.copy()
    result[found] = probability[found] * 10000.0 + (1 - probability[found]) * np.clip(literature[found], 0, 10000)
    return result


NN_COLUMNS = ["nn1_tanimoto", "nn1_si", "nn2_tanimoto", "nn2_si", "nn3_tanimoto", "nn3_si"]


def attach_nn(frame: pd.DataFrame) -> pd.DataFrame:
    """SI ближайших соседей и InChIKey14, которых нет в attach()."""
    table = pd.read_csv("data/processed/competition_open_data_lookup.csv",
                        usecols=["smiles", "inchikey14"]).drop_duplicates("smiles")
    frame = frame.merge(table.rename(columns={"smiles": "SMILES"}), on="SMILES", how="left")
    si_table = pd.read_csv("data/processed/competition_presence.csv",
                           usecols=["smiles", "nn1_si", "nn2_si", "nn3_si"]).drop_duplicates("smiles")
    return frame.merge(si_table.rename(columns={"smiles": "SMILES"}), on="SMILES", how="left")


def evaluate(frame: pd.DataFrame, y: np.ndarray, folds, seeds: int, av_mode: str,
             lookup_mode: str, alpha: float = 1.0) -> dict:
    n = len(y)
    prediction = np.full(n, np.nan)
    for fold in range(len(folds)):
        train, valid = fold_indices(folds, fold)
        probability, _, _ = predict_ensemble(frame.iloc[train], y[train], frame.iloc[valid], seeds, "hybrid")
        if av_mode == "constant":
            av = active_value_constant(frame.iloc[train], y[train])
        elif av_mode == "group":
            av = group_mean_model().fit(frame.iloc[train], y[train])
        elif av_mode == "ridge":
            av = ActiveValueModel(alpha=alpha).fit(frame.iloc[train], y[train])
        elif av_mode == "physical":
            av = PhysicalActiveValue().fit(frame.iloc[train], y[train])
        else:
            raise ValueError(av_mode)
        raw = probability * 10000.0 + (1 - probability) * av.predict(frame.iloc[valid])
        prediction[valid] = (hard_lookup(raw, frame.iloc[valid]) if lookup_mode == "hard"
                             else soft_lookup(raw, probability, frame.iloc[valid]))
    if not np.isfinite(prediction).all():
        raise ValueError("Incomplete predictions")
    return {"mse": float(np.mean((prediction - y) ** 2))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/active-value2"))
    args = parser.parse_args()
    data = load()
    frame, y = attach_nn(attach(data.train)), data.y
    from vacv.pipeline import scaffolds_of
    splits = {}
    for repeat in range(args.start_seed, args.start_seed + args.repeats):
        splits[f"random-{repeat}"] = [v for _, v in KFold(5, shuffle=True, random_state=repeat).split(y)]
    splits["scaffold"] = scaffold_folds(scaffolds_of(frame), n_splits=5)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, folds in splits.items():
        for av_mode in ("constant", "group", "ridge", "physical"):
            for lookup_mode in ("hard", "soft"):
                metrics = evaluate(frame, y, folds, args.seeds, av_mode, lookup_mode, args.alpha)
                results.append({"split": name, "av": av_mode, "lookup": lookup_mode, **metrics})
                print(f"{name:12} av={av_mode:8} lookup={lookup_mode:4} MSE={metrics['mse']:,.0f}", flush=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
