"""По-строчная оценка среднего SI активных: E[SI | соединение активно].

Глобальная константа «среднее активных» (285.7 на всей выборке) оставляет одну
строку с настоящим SI = 36066.667 с ошибкой порядка 10⁹ — примерно половину всей
MSE. Закон SI = CC50 / IC50 позволяет предсказывать активное значение по строке:

- известен IC50 (внешние базы, cytotox-записей нет): SI = CC50_hat / IC50, где
  CC50_hat — среднее измеренных CC50 похожих соединений, взвешенное по Tanimoto,
  стянутое к среднему CC50 той же клеточной линии;
- известен только CC50: SI = CC50 × (SI/CC50)_hat, где отношение берётся из
  похожих обучающих строк; среднее 1/IC50 по базе не годится — хвост сверхактивных
  соединений завышает его на два порядка;
- известны оба: SI = CC50 / IC50 напрямую;
- не известно ничего: сглаженное среднее активных по группе, затем глобальное.

Обучение — только активные строки outer-train соответствующего фолда, как у
остальных моделей. Записи базы, совпадающие с самой строкой по InChIKey14,
исключаются: их CC50 — это то, что оцениваем, а не признак.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .classify import extract_cell_line
from .pipeline import INACTIVE

ACTIVE_DB = Path("data/processed/vacv_si.csv")


@lru_cache(maxsize=8192)
def _fingerprint(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)


@lru_cache(maxsize=8192)
def _inchikey14(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return ""
    key = MolToInchiKey(mol)
    return key[:14] if key else ""


def _tanimoto_matrix(query, db) -> np.ndarray:
    from rdkit import DataStructs

    matrix = np.zeros((len(query), len(db)))
    for i, left in enumerate(query):
        if left is None:
            continue
        for j, right in enumerate(db):
            if right is not None:
                matrix[i, j] = DataStructs.TanimotoSimilarity(left, right)
    return matrix


@lru_cache(maxsize=1)
def _cc50_side() -> tuple[np.ndarray, list, list, list]:
    """Измеренные CC50 внешней базы: значения, отпечатки, скелеты, клеточные линии."""
    db = pd.read_csv(ACTIVE_DB)
    side = db[db["cc50_relation"].eq("=") & db["cc50"].notna()].drop_duplicates("inchikey14")
    values = side["cc50"].to_numpy(dtype=float)
    fingerprints = tuple(_fingerprint(s) for s in side["smiles"])
    keys = side["inchikey14"].fillna("").tolist()
    cells = side["cell_line"].astype(str).tolist()
    return values, list(fingerprints), keys, cells


class ActiveValueModel:
    """Условное среднее SI активных, оценённое по каждой строке отдельно."""

    def __init__(self, min_sim: float = 0.05, power: float = 2.0,
                 shrink: float = 2.0, train_power: float = 1.0,
                 fallback: str = "constant", fallback_shrink: float = 3.0):
        if min_sim < 0 or power < 0 or shrink < 0 or train_power < 0:
            raise ValueError("weights and thresholds must be non-negative")
        if fallback not in {"constant", "knn"}:
            raise ValueError("fallback must be constant or knn")
        self.min_sim = min_sim
        self.power = power
        self.shrink = shrink
        self.train_power = train_power
        self.fallback = fallback
        self.fallback_shrink = fallback_shrink

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "ActiveValueModel":
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("y must contain one finite non-negative target per row")
        self.train_frame_ = frame.reset_index(drop=True)
        self.train_y_ = y
        active = y != INACTIVE
        self.fallback_ = float(y[active].mean()) if active.any() else 0.0

        ic50 = pd.to_numeric(self.train_frame_["best_ic50_um"], errors="coerce").to_numpy(dtype=float)
        cc50 = pd.to_numeric(self.train_frame_["best_cc50_um"], errors="coerce").to_numpy(dtype=float)
        # Неявный CC50 = SI × IC50 там, где IC50 известен; иначе NaN.
        self.train_cc50_ = np.where(active, y, np.nan) * ic50
        # Отношение SI / CC50 там, где CC50 известен: обучающая цель для 1/IC50.
        ratio = np.full(len(frame), np.nan)
        known = active & np.isfinite(cc50) & (cc50 > 0)
        ratio[known] = y[known] / cc50[known]
        self.train_ratio_ = ratio

        values, fingerprints, keys, cells = _cc50_side()
        self.db_cc50_ = values
        self.db_fp_ = fingerprints
        self.db_keys_ = keys
        self.cc_prior_ = float(np.mean(values))
        self.cc_by_cell_: dict[str, float] = {}
        grouped = pd.DataFrame({"cell": cells, "cc": values}).groupby("cell")["cc"]
        for cell, group in grouped:
            if len(group) >= 5:
                self.cc_by_cell_[cell] = float(group.mean())

        self.train_fp_ = [_fingerprint(s) for s in self.train_frame_["SMILES"]]
        self.train_keys_ = [_inchikey14(s) for s in self.train_frame_["SMILES"]]
        return self

    def _cell_prior(self, cell: str | None) -> float:
        if cell and cell in self.cc_by_cell_:
            return self.cc_by_cell_[cell]
        return self.cc_prior_

    def _estimate(self, frame: pd.DataFrame, db_values: np.ndarray, train_values: np.ndarray,
                  prior: np.ndarray, use_db: bool = True) -> np.ndarray:
        """Взвешенное среднее значений похожих соединений, стянутое к prior."""
        query_fp = [_fingerprint(s) for s in frame["SMILES"]]
        query_keys = [_inchikey14(s) for s in frame["SMILES"]]
        db_fp = self.db_fp_ if use_db else self.train_fp_
        db_keys = self.db_keys_ if use_db else self.train_keys_
        sim = _tanimoto_matrix(query_fp, db_fp)
        train_sim = _tanimoto_matrix(query_fp, self.train_fp_)
        if np.isscalar(prior):
            prior = np.full(len(frame), float(prior))
        known = np.isfinite(train_values)
        train_prior = float(np.mean(train_values[known])) if known.any() else float(np.mean(prior))
        estimates = np.empty(len(frame))
        for i in range(len(frame)):
            weights = np.where(sim[i] > self.min_sim, sim[i] ** self.power, 0.0)
            if query_keys[i]:
                weights = weights * np.array([k != query_keys[i] for k in db_keys])
            train_w = np.where(known & (train_sim[i] > self.min_sim),
                               train_sim[i] ** self.train_power, 0.0)
            if query_keys[i]:
                train_w = train_w * np.array([k != query_keys[i] for k in self.train_keys_])
            total = weights.sum() + train_w.sum() + self.shrink
            if total <= self.shrink:
                estimates[i] = train_prior if known.any() else prior[i]
                continue
            db_part = float(np.dot(weights, db_values)) if weights.any() else 0.0
            train_part = float(np.nansum(train_w * np.nan_to_num(train_values, nan=0.0)))
            estimates[i] = (db_part + train_part + self.shrink * prior[i]) / total
        return estimates

    def _fallback_values(self, frame: pd.DataFrame) -> np.ndarray:
        """Оценка активного SI без внешних измерений: похожие активные строки обучения."""
        if self.fallback == "constant":
            return np.full(len(frame), self.fallback_)
        active = self.train_y_ != INACTIVE
        si = np.where(active, self.train_y_, np.nan)
        query_fp = [_fingerprint(s) for s in frame["SMILES"]]
        train_sim = _tanimoto_matrix(query_fp, self.train_fp_)
        query_keys = [_inchikey14(s) for s in frame["SMILES"]]
        result = np.empty(len(frame))
        for i in range(len(frame)):
            weights = np.where(np.isfinite(si) & (train_sim[i] > self.min_sim),
                               train_sim[i] ** self.power, 0.0)
            if query_keys[i]:
                weights = weights * np.array([k != query_keys[i] for k in self.train_keys_])
            total = weights.sum() + self.fallback_shrink
            if total <= self.fallback_shrink:
                result[i] = self.fallback_
                continue
            result[i] = (float(np.nansum(weights * np.nan_to_num(si, nan=0.0)))
                         + self.fallback_shrink * self.fallback_) / total
        return result

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        ic50 = pd.to_numeric(frame["best_ic50_um"], errors="coerce").to_numpy(dtype=float)
        cc50 = pd.to_numeric(frame["best_cc50_um"], errors="coerce").to_numpy(dtype=float)
        result = self._fallback_values(frame)
        ic_known = np.isfinite(ic50) & (ic50 > 0)
        cc_known = np.isfinite(cc50) & (cc50 > 0)
        both = ic_known & cc_known
        if both.any():
            result[both] = cc50[both] / ic50[both]
        ic_only = ic_known & ~both
        if ic_only.any():
            prior = np.array([self._cell_prior(extract_cell_line(c))
                              for c in frame.loc[ic_only, "Culture"]])
            cc_hat = self._estimate(frame[ic_only], self.db_cc50_, self.train_cc50_, prior)
            result[ic_only] = cc_hat / ic50[ic_only]
        cc_only = cc_known & ~ic_known
        if cc_only.any():
            known_ratio = self.train_ratio_[np.isfinite(self.train_ratio_)]
            ratio_prior = float(np.mean(known_ratio)) if len(known_ratio) else self.fallback_
            ratio_hat = self._estimate(frame[cc_only], self.train_ratio_, self.train_ratio_,
                                       ratio_prior, use_db=False)
            result[cc_only] = cc50[cc_only] * ratio_hat
        result = np.where(np.isfinite(result) & (result > 0), result, self.fallback_)
        return result
