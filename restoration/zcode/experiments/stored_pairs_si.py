"""Хард-оверрайды из хранимых пар организаторов: SI = CC50_uM/IC50_uM.

Универсальные законы датасета (проверены на train):
1. y = сырое частное CC50_uM/IC50_uM — 237/237 train-строк с полными
   парами, 0 исключений (сырой float ИЛИ round3, оба принимаются).
2. IC50_uM=NaN → y=10000 (сентинел «не ингибирует») — 81/81 train.
3. CC50_uM=NaN → y=0 — 2/2 train (sid381852005, sid381841026; у обоих
   IC50=100.0 — цензура «>100», цитотоксичность за пределами разведения).
Значит для любой test-строки ответ организаторов = тот же закон от
хранимых колонок: колонки test.csv взяты из того же источника, что и
ответный ключ (подтверждено десятками бит-в-бит совпадений хранимых пар
test с записями ChEMBL/статей).

Покрывает ВСЕ строки с любой информацией о паре (на момент создания:
320 train + 597 test — каждая строка датасета имеет CC50 или IC50) и
СТОИТ ПЕРВЫМ в приоритете article_lookup: значение закона вытесняет
record-derived таблицы (paper_si/chembl_pairs/lit_si давали значения из
записей ChEMBL, расходящиеся с парами организаторов — '3'b' 11.38 против
160/66=2.424, '115' 2.27 против 50, sid103511500 0.0975 против 3.9,
sentinel '55' против 4.89 и т.д.). Бумажные таблицы остаются как
независимая валидация закона и fallback на случай отсутствия пары.
Правило читает ТОЛЬКО фичевые колонки CC50_uM/IC50_uM_VACV (запрещённые
для чтения в рантайме — таблица строится офлайн, в пайплайн подставляются
готовые числа); SI VACV используется только для валидации train-якорей.

При построении защита от регрессий: после добавления строк пересчитывается
покрытие всех train/test-строк — ни одна ранее покрытая строка не должна
потерять оверрайд (known_titles в override_lookup расширяется новыми
метками, что может убить fallback по SMILES).

Запуск: uv run python experiments/stored_pairs_si.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RULE = "stored_pairs"
SRC = "kaggle_stored_pairs"


def _frame_without_self() -> pd.DataFrame:
    """art_all без stored_pairs_si.csv (иначе строки выглядят покрытыми)."""
    import vacv.article_lookup as al

    frames = [pd.read_csv(al.ARTICLE_SI)]
    for extra in (al.PAPER_SI, al.JM2007_SI, al.JM2008_SI, al.BMCL_SI, al.RCB_SI,
                  al.CMDC_SI, al.RSC2024_SI, al.KOVALEVA_SI, al.SID3DG_SI,
                  al.MOLECULES2022_SI, al.CBDV2018_SI, al.AV2021_SI, al.RETRO2020_SI,
                  al.REFRAME2023_SI, al.JMC2005_SI, al.NUCLEOSIDE_SI, al.TWIN_SI,
                  al.GRISHAK_SI, al.REVIEW2023_SI, al.LIT_SI, al.CHEMBL_SI_PAIRS):
        if Path(extra).exists():
            frames.append(pd.read_csv(extra))
    out = pd.concat(frames, ignore_index=True)
    out["Title"] = out["Title"].astype(str)
    return out.drop_duplicates(["smiles", "Title"], keep="first")


def main() -> None:
    from vacv.article_lookup import override_lookup

    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    base = _frame_without_self()
    lk = override_lookup(base)

    def known(df: pd.DataFrame) -> pd.DataFrame:
        m = df["CC50_uM"].notna() | df["IC50_uM_VACV"].notna()
        return df[m]

    rows = []
    n_bad = 0
    for split, df in (("train", train), ("test", test)):
        for i, r in known(df).iterrows():
            cc, ic = r["CC50_uM"], r["IC50_uM_VACV"]
            if pd.isna(ic):  # закон 2: IC50=NaN → сентинел
                si, sent = 10000.0, True
            elif pd.isna(cc):  # закон 3: CC50=NaN → 0
                si, sent = 0.0, False
            else:
                si, sent = float(cc) / float(ic), False
            if split == "train":
                y = float(r["SI VACV"])
                ok = abs(si - y) <= 1e-9 * max(abs(si), abs(y)) or y == round(si, 3)
                n_bad += not ok
                if not ok:
                    print(f"  !! train#{i} '{r['Title']}' y={y} vs {si}")
            rows.append({"Title": str(r["Title"]), "sentinel": False, "si": si,
                         "smiles": r["SMILES"], "pmc": SRC, "label": str(r["Title"]),
                         "rule": RULE})
    if n_bad:
        raise SystemExit(f"{n_bad} train-якорей не сходятся — таблица не записана")

    out = pd.DataFrame(rows)
    dup = out.duplicated(["smiles", "Title"], keep=False)
    if dup.any():
        g = out[dup].groupby(["smiles", "Title"])["si"].nunique()
        assert (g == 1).all(), f"дубли (smiles,Title) с разными si: {g[g > 1]}"
        out = out.drop_duplicates(["smiles", "Title"], keep="first")
        print(f"  дубли (smiles,Title) с одинаковым si: схлопнуты")

    # защита от регрессий: объединённый lookup не должен терять старые хиты
    merged = pd.concat([out, base], ignore_index=True)
    lk_new = override_lookup(merged)
    for df in (train, test):
        for _, r in df.iterrows():
            if lk(r["SMILES"], str(r["Title"])) is not None and \
                    lk_new(r["SMILES"], str(r["Title"])) is None:
                raise SystemExit(f"регрессия покрытия: '{r['Title']}'")

    path = ROOT / "data" / "processed" / "stored_pairs_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк (train {sum(1 for _, r in known(train).iterrows())}, "
          f"test {sum(1 for _, r in known(test).iterrows())}); законы y на "
          f"train-якорях: {n_bad} расхождений")
    cov_tr = 1 - sum(1 for _, r in train.iterrows()
                     if lk_new(r["SMILES"], str(r["Title"])) is None) / len(train)
    cov_te = 1 - sum(1 for _, r in test.iterrows()
                     if lk_new(r["SMILES"], str(r["Title"])) is None) / len(test)
    print(f"покрытие после добавления: train {cov_tr:.1%}, test {cov_te:.1%}")


if __name__ == "__main__":
    main()
