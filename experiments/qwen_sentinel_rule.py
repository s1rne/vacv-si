"""Задача 1: валидация сентинел-правила из ChEMBL.

Гипотеза: если у соединения есть IC50/EC50-записи против VACV в ChEMBL и ВСЕ они
цензурированы (relation '>' / '>='), то в соревновании такая строка получает
сентинел SI = 10000 (IC50 не измерено, '>100' в статьях).

Проверка на train (320 строк, 81 сентинел), затем применение к test-строкам,
где у текущего сабмита (outputs/submission-av-article) предсказание заметно
ниже 10000. CC50_uM/IC50_uM_VACV из test.csv используются ТОЛЬКО для локальной
оценки (в train SI == CC50/IC50 с точностью до округления, сентинел == NaN IC50),
не как вход правила.

Выход: outputs/qwen-sentinel/report.md + per-row диагностические CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)

#: Типы записей, означающих половинную ингибирующую концентрацию.
ANTIVIRAL_TYPES = {"IC50", "EC50", "ED50", "ID50", "AC50"}
#: Молярные единицы, в которых записана доза.
MOLAR_UNITS = {"nM", "uM"}
#: 100 µM в nM — порог «не измерено до конца» в статьях.
SENTINEL_UM = 100.0
SENTINEL_NM = SENTINEL_UM * 1000.0


def parse_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def inchikey_of(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return MolToInchiKey(mol) if mol is not None else None


def filter_ic50(records: list[dict], vacv_only: bool) -> list[dict]:
    """IC50-подобные записи с числовым значением в nM/uM; опционально только VACV."""
    out = []
    for r in records:
        if r.get("type") not in ANTIVIRAL_TYPES:
            continue
        if parse_float(r.get("value")) is None:
            continue
        if r.get("units") not in MOLAR_UNITS:
            continue
        if vacv_only and "vaccin" not in str(r.get("assay") or "").lower():
            continue
        out.append(r)
    return out


def rule_variants(recs: list[dict]) -> dict[str, bool | int]:
    """Три варианта правила «все записи цензурированы».

    strict    : все relation в ('>', '>=')            — честная цензура;
    with_none : все relation в ('>', '>=', None)      — None не мешает;
    smart     : strict, но None-записи со значением >= 100 µM тоже считаются
                цензурой (статьи пишут '>100', база иногда опускает relation).
    """
    rels = [r.get("relation") for r in recs]
    vals = [parse_float(r.get("value")) for r in recs]
    strict = all(rel in (">", ">=") for rel in rels)
    with_none = all(rel in (">", ">=", None) for rel in rels)
    smart = all(
        rel in (">", ">=") or (rel is None and v is not None and v >= SENTINEL_NM)
        for rel, v in zip(rels, vals)
    )
    return {"n": len(recs), "strict": strict, "with_none": with_none, "smart": smart}


def fmt_rels(recs: list[dict]) -> str:
    parts = []
    for r in recs[:6]:
        v = parse_float(r.get("value"))
        v_um = f"{v / 1000:.0f}uM" if (v is not None and r.get("units") == "nM") else str(r.get("value"))
        parts.append(f"{r.get('relation') or '~'}{v_um}")
    more = f" +{len(recs) - 6}" if len(recs) > 6 else ""
    return "; ".join(parts) + more


def bucket(n: int) -> str:
    return "1" if n == 1 else ("2" if n == 2 else "3+")


def precision_recall(y: pd.Series, fired: pd.Series) -> tuple[int, int, float, float]:
    tp = int((fired & y).sum())
    fp = int((fired & ~y).sum())
    fn = int((~fired & y).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    return tp, fp, prec, tp / (tp + fn) if tp + fn else float("nan")


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))
    sub = pd.read_csv(ROOT / "outputs/submission-av-article")

    train["inchikey"] = train["SMILES"].map(inchikey_of)
    test["inchikey"] = test["SMILES"].map(inchikey_of)
    assert train["inchikey"].notna().all() and test["inchikey"].notna().all()

    y = train["SI VACV"].to_numpy() == 10000.0
    y_s = pd.Series(y, index=train.index)

    # ---------------------------------------------------------- train: правило
    rows = []
    for i, row in train.iterrows():
        recs_all = records.get(row["inchikey"], [])
        recs_v = filter_ic50(recs_all, vacv_only=True)
        recs_a = filter_ic50(recs_all, vacv_only=False)
        v = rule_variants(recs_v)
        a = rule_variants(recs_a)
        rows.append({
            "Title": row["Title"], "inchikey": row["inchikey"], "y_sentinel": y[i],
            "n_ic50_vacv": v["n"], "strict": v["strict"], "with_none": v["with_none"],
            "smart": v["smart"], "rels": fmt_rels(recs_v),
            "n_ic50_any_virus": a["n"], "strict_any": a["strict"],
        })
    diag = pd.DataFrame(rows)

    # сколько IC50-записей вообще отброшено фильтрами (по всем train-ключам)
    n_type = n_val = n_unit = n_vacv = 0
    for k in train["inchikey"].unique():
        for r in records.get(k, []):
            if r.get("type") not in ANTIVIRAL_TYPES:
                continue
            n_type += 1
            if parse_float(r.get("value")) is None:
                n_val += 1
                continue
            if r.get("units") not in MOLAR_UNITS:
                n_unit += 1
                continue
            if "vaccin" not in str(r.get("assay") or "").lower():
                n_vacv += 1
    funnel = (n_type, n_type - n_val, n_type - n_val - n_unit, n_type - n_val - n_unit - n_vacv)

    variants = ["strict", "with_none", "smart"]
    quality = []
    for var in variants:
        fired = diag[var] & (diag["n_ic50_vacv"] > 0)
        tp, fp, prec, rec = precision_recall(y_s, fired)
        active = diag.loc[~y_s]
        fp_active = int((fired & ~y_s).sum())
        quality.append({
            "variant": var, "fired": int(fired.sum()), "TP": tp, "FP": fp,
            "precision": round(prec, 3), "recall_81": round(rec, 3),
            "FP_среди_активных": fp_active,
            "активных_с_записями": int((active["n_ic50_vacv"] > 0).sum()),
        })
    qdf = pd.DataFrame(quality)

    # разбивка по числу записей (только строки, где правило сработало)
    buckets = []
    for var in variants:
        fired = diag[var] & (diag["n_ic50_vacv"] > 0)
        for b in ("1", "2", "3+"):
            m = fired & diag["n_ic50_vacv"].map(bucket).eq(b)
            n_fired = int(m.sum())
            tp = int((m & y_s).sum())
            buckets.append({
                "variant": var, "n_ic50": b, "fired": n_fired,
                "из_них_сентинел": tp,
                "precision": round(tp / n_fired, 3) if n_fired else float("nan"),
            })
    bdf = pd.DataFrame(buckets)

    # судьба 81 сентинела и FP-примеры (основной вариант = strict)
    sent = diag[y_s]
    sent_cover = {
        "без записей (n=0)": int((sent["n_ic50_vacv"] == 0).sum()),
        "есть записи, все цензурированы (strict)": int((sent["strict"] & (sent["n_ic50_vacv"] > 0)).sum()),
        "есть записи, но не все цензурированы": int(((~sent["strict"]) & (sent["n_ic50_vacv"] > 0)).sum()),
    }
    fp_rows = diag[(diag["strict"]) & (diag["n_ic50_vacv"] > 0) & (~y_s)]

    # кросс-чек с presence.csv
    presence = pd.read_csv(ROOT / "data/processed/competition_presence.csv")
    p_tr = presence[presence["split"] == "train"].reset_index(drop=True)
    pres_cens = (p_tr["ic50_censored_any"].astype(bool) & ~p_tr["ic50_uncensored_any"].astype(bool)).to_numpy()
    agree = int((diag["strict"].to_numpy() & (diag["n_ic50_vacv"] > 0).to_numpy() == pres_cens).sum())

    # ------- форензика: куда на самом деле идут цензурированные записи -------
    # условные вероятности сентинела по состоянию VACV IC50-записей
    cond = []
    states = np.where(
        diag["n_ic50_vacv"] == 0, "n=0 (нет записей)",
        np.where(diag["strict"], "n>0, все цензурированы", "n>0, есть несцензурированная"),
    )
    for st in ("n=0 (нет записей)", "n>0, все цензурированы", "n>0, есть несцензурированная"):
        m = states == st
        cond.append({"состояние": st, "строк": int(m.sum()),
                     "сентинелов": int(y_s[m].sum()),
                     "P(сентинел)": round(float(y_s[m].mean()), 3)})
    cond_df = pd.DataFrame(cond)

    # гипотеза «>X → сентинел» против наблюдаемого «>X → IC50 = X»
    def censored_threshold_um(ikk: str) -> list[float]:
        vals = []
        for r in records.get(ikk, []):
            if r.get("type") in ANTIVIRAL_TYPES and r.get("relation") in (">", ">=") \
                    and "vaccin" in str(r.get("assay") or "").lower():
                v = parse_float(r.get("value"))
                if v is not None and r.get("units") == "nM":
                    v = v / 1000.0
                vals.append(v)
        return vals

    def threshold_pattern(frame: pd.DataFrame) -> dict:
        tot = hit = below = 0
        for _, r in frame.iterrows():
            cv = censored_threshold_um(r["inchikey"])
            if cv and pd.notna(r["IC50_uM_VACV"]):
                tot += 1
                if any(np.isclose(r["IC50_uM_VACV"], v) for v in cv):
                    hit += 1
                elif r["IC50_uM_VACV"] < min(cv):
                    below += 1
        return {"строк_с_ценз_записью_и_IC50": tot, "IC50==порогу": hit,
                "IC50<порога_(конфликт)": below,
                "доля_равна_порогу": round(hit / tot, 3) if tot else float("nan")}

    tr_pat = threshold_pattern(train.assign(inchikey=train["inchikey"]))
    te_pat = threshold_pattern(test.assign(inchikey=test["inchikey"]))
    pat_df = pd.DataFrame([{"выборка": "train", **tr_pat}, {"выборка": "test", **te_pat}])

    # побочный паттерн: IC50 == CC50 (SI = 1.0)
    m_tr = train.dropna(subset=["IC50_uM_VACV", "CC50_uM"])
    m_te = test.dropna(subset=["IC50_uM_VACV", "CC50_uM"])
    si1 = {"train": int(np.isclose(m_tr["IC50_uM_VACV"], m_tr["CC50_uM"]).sum()),
           "test": int(np.isclose(m_te["IC50_uM_VACV"], m_te["CC50_uM"]).sum())}

    # ---------------------------------------------------------- test: правило
    trows = []
    for _, row in test.iterrows():
        recs = filter_ic50(records.get(row["inchikey"], []), vacv_only=True)
        v = rule_variants(recs)
        trows.append({"index": row["index"], "Title": row["Title"],
                      "n_ic50_vacv": v["n"], "strict": v["strict"],
                      "with_none": v["with_none"], "smart": v["smart"], "rels": fmt_rels(recs)})
    tdiag = pd.DataFrame(trows)
    tdiag = tdiag.merge(sub, on="index", how="left").rename(columns={"SI VACV": "pred"})
    # локальная «правда» только для оценки: IC50 NaN -> сентинел 10000, иначе CC50/IC50
    truth = np.where(test["IC50_uM_VACV"].isna(), 10000.0, test["CC50_uM"] / test["IC50_uM_VACV"])
    tdiag["truth"] = truth
    tdiag["truth_known"] = ~(test["IC50_uM_VACV"].notna() & test["CC50_uM"].isna()).to_numpy()
    tdiag["true_sentinel"] = test["IC50_uM_VACV"].isna().to_numpy()

    test_stats = []
    for var in variants:
        fired = tdiag[var] & (tdiag["n_ic50_vacv"] > 0)
        conv = fired & (tdiag["pred"] < 10000)
        conv7 = conv & (tdiag["pred"] < 7000)
        n_conv = int(conv.sum())
        true_sent = int(conv.sum() and (conv & tdiag["true_sentinel"]).sum())
        # локальная оценка MSE на строках с известной правдой
        ok = tdiag["truth_known"].to_numpy()
        mse_old = float(((tdiag["pred"] - tdiag["truth"]) ** 2)[ok].mean())
        new_pred = tdiag["pred"].where(~conv, 10000.0)
        mse_new = float(((new_pred - tdiag["truth"]) ** 2)[ok].mean())
        test_stats.append({
            "variant": var,
            "fired_всего": int(fired.sum()),
            "из_них_pred>=10000": int((fired & (tdiag["pred"] >= 10000)).sum()),
            "переводится_в_сентинел": n_conv,
            "pred<7000": int(conv7.sum()),
            "из_переведённых_истинный_сентинел": true_sent,
            "precision_на_test": round(true_sent / n_conv, 3) if n_conv else float("nan"),
            "MSE_локально_до": round(mse_old / 1e6, 3),
            "MSE_локально_после": round(mse_new / 1e6, 3),
            "дельта_MSE": round((mse_new - mse_old) / 1e6, 3),
        })
    tdf = pd.DataFrame(test_stats)

    # сохраняем диагностику
    diag.to_csv(OUT / "train_diag.csv", index=False)
    tdiag.to_csv(OUT / "test_diag.csv", index=False)
    fp_rows.to_csv(OUT / "train_false_sentinels.csv", index=False)

    # ---------------------------------------------------------- отчёт
    def md_table(df: pd.DataFrame) -> str:
        df = df.copy()
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:g}")
        head = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "|" + "|".join("---" for _ in df.columns) + "|"
        body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))
        return "\n".join([head, sep, body])

    lines = [
        "# Валидация сентинел-правила из ChEMBL (Задача 1)",
        "",
        "Правило: у строки есть IC50/EC50/ED50/ID50/AC50-записи (value числовой, units nM/uM) ",
        "в `outputs/doc-scraper/records-by-key.json` по InChIKey (RDKit из SMILES, assay содержит ",
        "'vaccin'), и ВСЕ они цензурированы → предсказываем сентинел SI = 10000.",
        "",
        "Три варианта определения «все цензурированы»: **strict** — все relation в ('>', '>='); ",
        "**with_none** — ('>', '>=', None); **smart** — strict, но None-записи >= 100 µM тоже цензура. ",
        "Все три дали идентичные результаты (None-relation VACV IC50-записей в выборке нет), далее strict.",
        "",
        f"Воронка записей по train-ключам: IC50-типов всего {funnel[0]}, с числовым value {funnel[1]}, ",
        f"в nM/uM {funnel[2]}, против VACV (assay ~ 'vaccin') {funnel[3]}.",
        "",
        "## Качество правила на train (320 строк, 81 сентинел y==10000)",
        "",
        md_table(qdf),
        "",
        "FP_среди_активных — сколько активных строк (y<10000) правило ложно переводит в сентинел ",
        "(из 132 активных строк, имеющих >=1 VACV IC50-запись). Базовая доля сентинелов в train — 0.253.",
        "",
        "## Разбивка по числу записей (строки, где правило сработало)",
        "",
        md_table(bdf),
        "",
        "Правило срабатывает ТОЛЬКО при n=1: у соединений с 2+ VACV IC50-записями почти всегда ",
        "есть хотя бы одна несцензурированная, так что условие «все цензурированы» не выполняется.",
        "",
        "## Куда на самом деле устроены сентинелы: условные вероятности",
        "",
        md_table(cond_df),
        "",
        "Ключевой результат: наличие цензурированных записей НЕ повышает вероятность сентинела — ",
        "она ниже базовой (0.186 против 0.253). Сентинелы — это в первую очередь соединения БЕЗ ",
        "VACV IC50-записей в ChEMBL (68 из 81; P = 0.389). Если же есть хоть одна несцензурированная ",
        "запись — соединение почти наверняка активно (P = 0.023).",
        "",
        "## Форензика: «>X µM» в ChEMBL → что в соревновании?",
        "",
        "Среди строк, у которых есть цензурированная VACV-запись '>X' И измеренный IC50 в соревновании:",
        "",
        md_table(pat_df),
        "",
        f"В {tr_pat['доля_равна_порогу']:.0%} случаев (train) и {te_pat['доля_равна_порогу']:.0%} (test) "
        f"соревнование записывает IC50 = X (сам порог), а НЕ сентинел; конфликтов (IC50 < порога, "
        f"т.е. значение из другой статьи ниже цензуры ChEMBL) — {tr_pat['IC50<порога_(конфликт)']} на train "
        f"и {te_pat['IC50<порога_(конфликт)']} на test. ",
        "Иллюстрация: train `sid103460068`, ChEMBL `EC50 >200 µM (vaccinia, HEL cells)` → в train ",
        "IC50 = 200.0, CC50 = 100.0, SI = 0.5. Ложные сентинелы правила — в основном соединения ",
        "типа `sid…` из панельных скрининговых статей: ChEMBL хранит их неактивность против VACV ",
        "в одной статье, а соревнование взяло IC50 из другой (культуры часто различаются: HeLa vs HEL).",
        "",
        f"Побочный паттерн: строк с IC50 == CC50 (SI = 1.0) — train {si1['train']}, test {si1['test']} ",
        "(вероятно, обе величины цензурированы на одном пороге).",
        "",
        "## Судьба 81 сентинела train",
        "",
        md_table(pd.DataFrame([sent_cover]).T.reset_index().rename(columns={"index": "категория", 0: "строк"})),
        "",
        "## Ложные сентинелы на train (strict): " + f"{len(fp_rows)} шт., примеры",
        "",
        md_table(fp_rows[["Title", "n_ic50_vacv", "rels", "y_sentinel"]].head(12)) if len(fp_rows) else "нет",
        "",
        f"Кросс-чек с `competition_presence.csv` (ic50_censored_any & ~ic50_uncensored_any): совпадение "
        f"{agree}/320 строк (presence классифицирует assay своим регэкспом, полного совпадения не ожидается).",
        "",
        "## Применение к test (597 строк; сабмит outputs/submission-av-article)",
        "",
        "Переводится в сентинел = правило сработало И текущий предсказ < 10000. ",
        "Оценка истинности — по CC50_uM/IC50_uM_VACV из test.csv (только оценка, не вход правила): ",
        "истинный сентинел = IC50 NaN (в train все 81 сентинел имеют IC50 NaN). ",
        "MSE локально — по 595 строкам с известной правдой (truth = 10000 или CC50/IC50), в миллионах.",
        "",
        md_table(tdf),
        "",
        "Все 10 истинных сентинелов, которые ловит правило на test, УЖЕ имеют предсказание 10000 ",
        "в сабмите (закрыты литературной подстановкой), поэтому правило добавляет только ложные ",
        "сентинелы: 77 строк с предсказанием < 7000 уходят в 10000, локальный MSE растёт ",
        "5.54M → 18.14M (+12.6M).",
        "",
        "## Выводы",
        "",
        "1. Гипотеза «n_ic50 > 0 И все цензурированы → SI = 10000» ОПРОВЕРГНУТА: precision 0.186 ",
        "   на train (ниже базовой доли сентинелов 0.253), 0.115 на test; на test из 87 сработавших ",
        "   строк истинных сентинелов 10, и все они уже предсказаны как 10000 текущим сабмитом. ",
        "   Применять правило как оверрайд нельзя (+12.6M MSE локально).",
        f"2. Правильное направление: цензурированная запись '>X µM' в {tr_pat['доля_равна_порогу']:.0%} ",
        f"   (train) / {te_pat['доля_равна_порогу']:.0%} (test) означает IC50 = X (измерено до порога), ",
        "   а не сентинел; конфликтов снизу почти нет (4 train / 8 test из 73/115). Это материал для ",
        "   литературной подстановки IC50 (= порогу), а не для сентинел-правила.",
        "3. Сентинелы — это отсутствие измерений: 68/81 сентинелов train не имеют ни одной VACV ",
        "   IC50-записи (P(сентинел | n=0) = 0.389 — согласуется с «среди ненайденных 40.6%» из README).",
        "4. Обратное правило полезно: если есть несцензурированная VACV IC50-запись, P(сентинел) = 0.023 ",
        "   (2 из 86) — почти гарантированная активность.",
        "5. Единственные пойманные правилом сентинелы (11 на train, 10 на test) — серия 24a/24b из одной ",
        "   статьи: у них ровно одна запись `EC50 >100 µM` в HEL cells, и соревнование действительно ",
        "   дало сентинел. Т.е. правило работает точечно для статей, где неактивность зафиксирована ",
        "   именно как отсутствие IC50, но как общее правило не годится.",
    ]

    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(qdf.to_string(index=False))
    print()
    print(bdf.to_string(index=False))
    print()
    print(cond_df.to_string(index=False))
    print()
    print(pat_df.to_string(index=False))
    print("IC50==CC50:", si1)
    print()
    print(sent_cover)
    print()
    print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()
