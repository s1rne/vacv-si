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
        "в `outputs/doc-scraper/records-by-key.json` по InChIKey (RDKit из SMILES), и ВСЕ они ",
        "цензурированы → предсказываем сентинел SI = 10000.",
        "",
        "Три варианта определения «все цензурированы»:",
        "- **strict** — все relation в ('>', '>=');",
        "- **with_none** — все relation в ('>', '>=', None) (None не мешает);",
        "- **smart** — strict, но None-записи со значением >= 100 µM тоже цензура.",
        "",
        f"IC50-фильтр по assay содержит 'vaccin' (VACV-специфичные записи). Воронка по train-ключам: "
        f"записей IC50-типов {funnel[0]}, с числовым value {funnel[1]}, в nM/uM {funnel[2]}, "
        f"против VACV {funnel[3]}.",
        "",
        "## Качество на train (320 строк, 81 сентинел y==10000)",
        "",
        md_table(qdf),
        "",
        "FP_среди_активных — сколько активных строк (y<10000) правило ложно переводит в сентинел; ",
        "последний столбец — сколько активных строк вообще имеют >=1 VACV IC50-запись (знаменатель риска).",
        "",
        "## Разбивка по числу записей (строки, где правило сработало)",
        "",
        md_table(bdf),
        "",
        "## Судьба 81 сентинела train (variant strict)",
        "",
        md_table(pd.DataFrame([sent_cover]).T.reset_index().rename(columns={"index": "категория", 0: "строк"})),
        "",
        "## Ложные сентинелы на train (strict)",
        "",
        f"Всего {len(fp_rows)}. Примеры:",
        "",
        md_table(fp_rows[["Title", "n_ic50_vacv", "rels", "y_sentinel"]].head(12)) if len(fp_rows) else "нет",
        "",
        "Кросс-чек с `competition_presence.csv` (ic50_censored_any & ~ic50_uncensored_any): совпадение "
        f"{agree}/320 строк ( presence считает по своей классификации assay, совпадение не обязано быть полным ).",
        "",
        "## Применение к test (597 строк; сабмит outputs/submission-av-article)",
        "",
        "Переводится в сентинел = правило сработало И текущий предсказ < 10000. ",
        "Оценка истинности — по CC50_uM/IC50_uM_VACV из test.csv (только оценка, не вход правила): ",
        "истинный сентинел = IC50 NaN (в train все 81 сентинел имеют IC50 NaN). ",
        "MSE локально — по 597-2 строкам с известной правдой (truth = 10000 или CC50/IC50), в миллионах.",
        "",
        md_table(tdf),
        "",
        "## Выводы",
        "",
    ]

    best = qdf.sort_values("precision", ascending=False).iloc[0]
    lines.append(
        f"- Лучший по precision вариант: **{best['variant']}** "
        f"(precision {best['precision']}, recall {best['recall_81']}, FP {best['FP']})."
    )
    if len(fp_rows):
        lines.append(f"- Ложные сентинелы train (strict): {len(fp_rows)} шт., примеры выше — стоит посмотреть, "
                     "не «тестили в другой статье против другого вируса» ли это.")
    lines.append(f"- На test правило переводит в сентинел "
                 f"{int(tdf.iloc[0]['переводится_в_сентинел'])} строк (strict) при "
                 f"{int(tdf.iloc[0]['pred<7000'])} из них с предсказанием < 7000.")

    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(qdf.to_string(index=False))
    print()
    print(bdf.to_string(index=False))
    print()
    print(sent_cover)
    print()
    print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()
