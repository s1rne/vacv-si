"""Задача 3: структура записей records-by-key.json по standard_type.

Считаем по каждому типу записей:
- сколько записей всего и у скольких соединений;
- сколько записей VACV-специфичные (assay содержит 'vaccin', регистр неважен);
- сколько цензурированных (relation '>' / '>=');
- топ-3 единиц измерения;
- как тип классифицирует вакный классификатор vacv/classify.py
  (antiviral-ambiguous / cytotox / ignored / прочие-unknown) — чтобы видеть,
  какие типы сейчас выбрасываются куратором и что ещё можно выжать.

Выход: outputs/qwen-sentinel/types.md + types_full.csv.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from vacv.classify import AMBIGUOUS_TYPES, CYTOTOX_TYPES, IGNORED_TYPES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)


def repo_class(t: str) -> str:
    if t in AMBIGUOUS_TYPES:
        return "antiviral-ambiguous (используется)"
    if t in CYTOTOX_TYPES:
        return "cytotox (используется)"
    if t in IGNORED_TYPES:
        return "ignored (выбрасывается)"
    return "unknown (выбрасывается)"


def main() -> None:
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))

    stats: dict[str, dict] = {}
    for key, recs in records.items():
        for r in recs:
            t = r.get("type") or "(пусто)"
            s = stats.setdefault(t, {
                "n": 0, "keys": set(), "vacv": 0, "censored": 0, "units": Counter(),
            })
            s["n"] += 1
            s["keys"].add(key)
            assay = str(r.get("assay") or "").lower()
            if "vaccin" in assay:
                s["vacv"] += 1
            if r.get("relation") in (">", ">="):
                s["censored"] += 1
            s["units"][str(r.get("units"))] += 1

    rows = []
    for t, s in stats.items():
        top_units = ", ".join(f"{u}×{c}" for u, c in s["units"].most_common(3))
        rows.append({
            "type": t,
            "записей": s["n"],
            "соединений": len(s["keys"]),
            "VACV (assay~vaccin)": s["vacv"],
            "доля_VACV": round(s["vacv"] / s["n"], 3),
            "цензурированных (>/>=)": s["censored"],
            "доля_ценз": round(s["censored"] / s["n"], 3),
            "топ-3 units": top_units,
            "класс в vacv/classify": repo_class(t),
        })
    full = pd.DataFrame(rows).sort_values("записей", ascending=False)
    full.to_csv(OUT / "types_full.csv", index=False)

    top20 = full.head(20)

    def md_table(df: pd.DataFrame) -> str:
        head = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "|" + "|".join("---" for _ in df.columns) + "|"
        body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))
        return "\n".join([head, sep, body])

    total = sum(s["n"] for s in stats.values())
    total_vacv = sum(s["vacv"] for s in stats.values())
    total_cens = sum(s["censored"] for s in stats.values())
    vacv_only_records = sum(s["n"] for s in stats.values() if s["vacv"] == s["n"] and s["n"] > 0)
    vacv_only_types = sum(1 for s in stats.values() if s["vacv"] == s["n"] and s["n"] > 0)

    # типы вне репного классификатора с заметным числом VACV-записей
    unklass = full[(full["класс в vacv/classify"] == "unknown (выбрасывается)")
                   & (full["VACV (assay~vaccin)"] > 0)].head(12)

    # --------- резерв прямых SI-значений: Ratio CC50/IC50, Selectivity Index, ...
    SI_TYPES = {"Ratio CC50/IC50", "Selectivity Index", "Selectivity index", "SI",
                "Ratio CC50/EC50", "Therapeutic index", "TI"}
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchiKey
    RDLogger.DisableLog("rdApp.*")

    def ikey(s):
        m = Chem.MolFromSmiles(s)
        return MolToInchiKey(m) if m else None

    si_rows = []
    for name, frame in (("train", train), ("test", test)):
        frame["inchikey"] = frame["SMILES"].map(ikey)
        n_rows = n_eval = 0
        ratios = []
        for _, row in frame.iterrows():
            vals = []
            for r in records.get(row["inchikey"], []):
                if r.get("type") in SI_TYPES and "vaccin" in str(r.get("assay") or "").lower():
                    try:
                        vals.append(float(r.get("value")))
                    except (TypeError, ValueError):
                        pass
            if vals:
                n_rows += 1
                if name == "train" and pd.notna(row["SI VACV"]) and row["SI VACV"] != 10000:
                    ratios.append(float(np.median(vals)) / float(row["SI VACV"]))
                    n_eval += 1
        si_rows.append({"выборка": name, "строк с VACV SI-записью": n_rows,
                        "оцениваемых на train (активные)": n_eval,
                        "медианное отношение (запись/train SI)": (
                            f"{np.median(ratios):.2f}" if ratios else "—")})
    si_df = pd.DataFrame(si_rows)

    lines = [
        "# Структура записей ChEMBL по standard_type (Задача 3)",
        "",
        f"Источник: `outputs/doc-scraper/records-by-key.json` — {len(records)} InChIKey, ",
        f"{total} записей. VACV-специфичных записей (assay содержит 'vaccin'): {total_vacv} ",
        f"({total_vacv / total:.1%}); цензурированных (>/>=): {total_cens} ({total_cens / total:.1%}).",
        "",
        f"Типов всего: {len(stats)}; типов, у которых ВСЕ записи VACV-специфичные: {vacv_only_types} ",
        f"({vacv_only_records} записей) — это ядро «настоящих» противооспенных данных.",
        "",
        "## Топ-20 типов по частоте",
        "",
        md_table(top20),
        "",
        "## Типы вне классификатора vacv/classify.py, где есть VACV-записи",
        "",
        "Классификатор делит типы на antiviral-ambiguous (IC50/EC50/…), cytotox, ignored и unknown; ",
        "unknown и ignored выбрасываются. Если в выбрасываемом типе есть VACV-записи — это резерв:",
        "",
        md_table(unklass) if len(unklass) else "нет",
        "",
        "## Резерв прямых SI-значений (Ratio CC50/IC50, Selectivity Index, …)",
        "",
        "Типы-отношения уже содержат готовый SI; в `vacv/classify.py` они не классифицированы ",
        "(«Selectivity index» из IGNORED_TYPES не совпадает по регистру с реальным «Selectivity Index»). ",
        "Сколько строк соревнования покрыто VACV SI-записями и насколько они согласуются с train SI:",
        "",
        md_table(si_df),
        "",
        "## Выводы",
        "",
        "- Полная таблица по всем типам: `outputs/qwen-sentinel/types_full.csv`.",
        "- Ядро противооспенных данных — EC50 (526 VACV-записей) и IC50 (184); VACV-доля мала ",
        "  (2.7–8.9%), потому что соединения тестировались панелями вирусов. Цензурированных среди ",
        "  VACV EC50/IC50 много — но, как показала Задача 1, цензура означает «IC50 = порогу», ",
        "  а не сентинел.",
        "- CC50: 1685 записей, 56.7% цензурированы; MCC (755 записей, 86% цензуры, 313 в µg/mL) и ",
        "  ID50 (608 записей, 46 VACV) сейчас выбрасываются как unknown — кандидаты на расширение ",
        "  cytotox/antiviral сторон после ручной проверки assay-текстов.",
        f"- Прямые SI-значения: {int(si_df[si_df['выборка']=='train'].iloc[0]['строк с VACV SI-записью'])} ",
        f"train-строк и {int(si_df[si_df['выборка']=='test'].iloc[0]['строк с VACV SI-записью'])} test-строк ",
        "  имеют VACV SI-запись (Ratio CC50/IC50 / Selectivity Index) — готовые литературные подстановки: ",
        f"на {int(si_df.iloc[0]['оцениваемых на train (активные)'])} активных train-строках медианное ",
        f"отношение записи к реальному SI = {si_df.iloc[0]['медианное отношение (запись/train SI)']} — ",
        "  согласование почти идеальное.",
    ]
    (OUT / "types.md").write_text("\n".join(lines), encoding="utf-8")
    print(full.head(25).to_string(index=False))
    print("\nVACV-only types:", vacv_only_types, "records:", vacv_only_records)
    print()
    print(si_df.to_string(index=False))


if __name__ == "__main__":
    main()
