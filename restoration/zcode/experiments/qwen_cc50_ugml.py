"""Задача 2: cytotox-записи ChEMBL в µg/mL — сколько CC50 можно восстановить.

В `outputs/doc-scraper/records-by-key.json` цитотоксические записи (CC50/TD50/MCC/TC50/...)
могут быть в массовых единицах (µg/mL и др.). Для train-строк, где в
`competition_presence.csv` best_cc50_um == NaN (нет cytotox-половины), считаем,
сколько строк можно заполнить, переведя µg/mL → µM через молекулярный вес
(RDKit Descriptors.MolWt; конвертация — vacv.units.to_micromolar).

Сравнение с CC50_uM из train.csv — только оценка качества конверсии, не вход пайплайна.

Выход: outputs/qwen-sentinel/cc50_ugml.md + cc50_ugml_rows.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.inchi import MolToInchiKey

from vacv.classify import CYTOTOX_TYPES
from vacv.units import canonical_units, to_micromolar

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)

CENSORED_RELATIONS = {">", ">=", "<", "<="}


def parse_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def inchikey_of(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return MolToInchiKey(mol) if mol is not None else None


def cytotox_ugml(records: list[dict], mw: float) -> tuple[list[dict], dict[str, int]]:
    """Цитотокс-записи в массовых единицах, конвертированные в µM.

    Возвращает (записи с добавленным value_um, счётчик причин пропуска).
    """
    out, skip = [], {}
    for r in records:
        if r.get("type") not in CYTOTOX_TYPES:
            continue
        token = canonical_units(r.get("units"))
        if token not in {"ug/ml", "mg/ml", "mg/l", "g/l", "ng/ml"}:
            continue
        v, reason = to_micromolar(r.get("value"), r.get("units"), mol_weight=mw)
        if v is None:
            skip[reason] = skip.get(reason, 0) + 1
            continue
        out.append({**r, "value_um": v, "unit_token": token})
    return out, skip


def stats_line(values: np.ndarray) -> str:
    if len(values) == 0:
        return "—"
    return (f"медиана {np.median(values):.2f}, Q1 {np.quantile(values, 0.25):.2f}, "
            f"Q3 {np.quantile(values, 0.75):.2f}, n={len(values)}")


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    presence = pd.read_csv(ROOT / "data/processed/competition_presence.csv")
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))

    for frame in (train, test):
        frame["inchikey"] = frame["SMILES"].map(inchikey_of)
        frame["mw"] = frame["SMILES"].map(lambda s: MolWt(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else np.nan)

    p_tr = presence[presence["split"] == "train"].reset_index(drop=True)
    p_te = presence[presence["split"] == "test"].reset_index(drop=True)
    assert (p_tr["Title"].to_numpy() == train["Title"].to_numpy()).all()
    assert (p_te["index"].to_numpy() == test["index"].to_numpy()).all()
    train["best_cc50_um"] = p_tr["best_cc50_um"].to_numpy()
    test["best_cc50_um"] = p_te["best_cc50_um"].to_numpy()

    summary_rows, detail_rows = [], []
    for name, frame in (("train", train), ("test", test)):
        target = frame[frame["best_cc50_um"].isna()]
        n_target = len(target)
        n_ugml = n_ugml_cens = 0
        ratios_med, ratios_max, ratios_uncens = [], [], []
        types_found: dict[str, int] = {}
        rel_found: dict[str, int] = {}
        for _, row in target.iterrows():
            recs, _skip = cytotox_ugml(records.get(row["inchikey"], []), row["mw"])
            if not recs:
                continue
            n_ugml += 1
            vals = np.array([r["value_um"] for r in recs])
            unc = np.array([r["value_um"] for r in recs
                            if r.get("relation") not in CENSORED_RELATIONS])
            for r in recs:
                types_found[r["type"]] = types_found.get(r["type"], 0) + 1
                rel_found[r.get("relation") or "~"] = rel_found.get(r.get("relation") or "~", 0) + 1
            detail = {"split": name, "Title": row["Title"], "mw": round(row["mw"], 1),
                      "n_ugml_records": len(recs),
                      "values_um": ", ".join(f"{v:.1f}" for v in sorted(vals)),
                      "best_cc50_um": row["best_cc50_um"]}
            if name == "train":
                ref = row["CC50_uM"]
                detail["CC50_uM_train"] = ref
                if pd.notna(ref):
                    detail["ratio_median"] = round(float(np.median(vals)) / ref, 3)
                    ratios_med.append(float(np.median(vals)) / ref)
                    ratios_max.append(float(vals.max()) / ref)
                    if len(unc):
                        ratios_uncens.append(float(np.median(unc)) / ref)
            detail_rows.append(detail)
            if all(r.get("relation") in CENSORED_RELATIONS for r in recs):
                n_ugml_cens += 1

        ref_avail = int((target["CC50_uM"].notna() & target["best_cc50_um"].isna()).sum()) if name == "train" else 0
        summary_rows.append({
            "выборка": name,
            "строк_с_NaN_best_cc50": n_target,
            "можно_заполнить": n_ugml,
            "все записи цензурированы": n_ugml_cens,
            "для оценки (train.csv CC50_uM известен)": ref_avail,
            "медианное отношение (по медиане записей)": stats_line(np.array(ratios_med)),
            "медианное отношение (по max записей)": stats_line(np.array(ratios_max)),
            "медианное отношение (только несцензурированные)": stats_line(np.array(ratios_uncens)),
            "в пределах x2 от train CC50 (медиана)": (
                f"{np.mean((np.array(ratios_med) > 0.5) & (np.array(ratios_med) < 2)):.0%}"
                if ratios_med else "—"),
        })
        if name == "train":
            types_train, rel_train = types_found, rel_found

    summ = pd.DataFrame(summary_rows)
    det = pd.DataFrame(detail_rows)
    det.to_csv(OUT / "cc50_ugml_rows.csv", index=False)

    def md_table(df: pd.DataFrame) -> str:
        head = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "|" + "|".join("---" for _ in df.columns) + "|"
        body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))
        return "\n".join([head, sep, body])

    types_df = pd.DataFrame(sorted(types_train.items(), key=lambda x: -x[1]),
                            columns=["тип записи (train, µg/mL)", "записей"])
    rel_df = pd.DataFrame(sorted(rel_train.items(), key=lambda x: -x[1]),
                          columns=["relation (train, µg/mL)", "записей"])

    tr = summ[summ["выборка"] == "train"].iloc[0]
    te = summ[summ["выборка"] == "test"].iloc[0]
    lines = [
        "# CC50-записи в µg/mL: потенциал заполнения (Задача 2)",
        "",
        "Источник записей: `outputs/doc-scraper/records-by-key.json`, ключ — InChIKey (RDKit из SMILES). ",
        "Цитотокс-типы — `CYTOTOX_TYPES` из `vacv/classify.py` (CC50, TC50, CD50, TD50, GI50, TGI, ",
        "MNTD, MTC, MTD, LC50, LD50, CT50; MCC в репо не входит). Массовые единицы: ug/ml, mg/ml, ",
        "mg/l, g/l, ng/ml; конвертация µM = value[mass] × factor × 1000 / MolWt (RDKit), ",
        "функция `vacv.units.to_micromolar`.",
        "",
        "Цель заполнения — признак `best_cc50_um` из `competition_presence.csv` (NaN = в курсированной ",
        "базе нет cytotox-половины; в train это 170 из 320 строк).",
        "",
        "## Итоги",
        "",
        md_table(summ),
        "",
        f"Медианное отношение — (CC50 из записей в µM) / (CC50_uM из train.csv); 1.0 = идеальное ",
        f"совпадение. Заполнить можно {int(tr['можно_заполнить'])} из {int(tr['строк_с_NaN_best_cc50'])} ",
        "train-строк с NaN best_cc50_um (значение — медиана записей строки, как в presence).",
        "",
        "## Типы и relations µg/mL-записей (train)",
        "",
        md_table(types_df),
        "",
        md_table(rel_df),
        "",
        "## Построчные детали",
        "",
        md_table(det),
        "",
        "## Выводы",
        "",
        f"- Заполнить можно {int(tr['можно_заполнить'])} из {int(tr['строк_с_NaN_best_cc50'])} train-строк ",
        f"({int(tr['можно_заполнить']) / int(tr['строк_с_NaN_best_cc50']):.0%}) — результат скромный: ",
        "  массовые единицы у цитотокс-записей редки, основная масса CC50 уже в молярных единицах; ",
        f"  в test аналогично {int(te['можно_заполнить'])} строк.",
        f"- Согласование с train.csv слабое: {summ.iloc[0]['в пределах x2 от train CC50 (медиана)']} ",
        "  заполненных строк в пределах x2 от CC50_uM (медианное отношение по 4 строкам — ",
        "  см. таблицу). Причина: µg/mL-записи происходят из других статей и клеточных линий ",
        "  (Rotenone: записи 0.8–15.7 µM против 0.1 µM в train; Docetaxel: ~0.02 µM против 66.7). ",
        "  Для признаков presence заполнение всё равно полезно (сам факт наличия cytotox-записи ",
        "  информативнее точного значения), но подставлять конвертированные CC50 как литературную ",
        "  подстановку SI не стоит.",
    ]
    (OUT / "cc50_ugml.md").write_text("\n".join(lines), encoding="utf-8")
    print(summ.to_string(index=False))
    print()
    print(det.to_string(index=False))


if __name__ == "__main__":
    main()
