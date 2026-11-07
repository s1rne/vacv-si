"""Задача 6: прямые SI-записи других типов (Ratio CC50/EC50, Ratio EC50, Ratio IC50,
SI, Therapeutic index + Selectivity Index) — валидация пар (type, doc) и предложение
расширения candidate() из experiments/lit_si_rule.py.

Методология (требование управляющего агента, эталон experiments/lit_si_cv.py):
правило (type, doc) валидируется только по разрешённой целевой колонке train и
полностью строится ВНУТРИ фолда: fit=4/5 находит валидированные пары, применение
к valid=1/5, оценка — сохранённые OOF-прокси через piece_mse. Полно-трейновые
списки пар — только диагностика в отчёте.

Правило-кандидат (спецификация задания + конфликт-правило lit_si_rule.candidate):
  - запись типа-отношения, assay ~ 'vaccin', числовое значение > 0;
  - строка годна, если нет VACV-полной пары (CC50+несценз. VACV IC50) ни в одном
    доке (соревнование могло взять пару оттуда) и нет прямого production-Ratio
    (Rule A уже решает строку; при его неоднозначности строку не трогаем);
  - все записи (type, doc)-кандидата по всем fit-строкам бит-в-бит (допуск 0.3%)
    равны y — иначе пара бракуется; минимум MIN_WITNESS согласованных fit-строк;
  - на valid применяется только если все кандидат-значения строки согласованы
    (0.5%), строка не покрыта таблицами статей и не имеет production-оверрайда.

Выход: outputs/qwen-sentinel/ratio_types_cv.csv + ratio_types.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.active_value2 import attach_nn
from experiments.lit_si_cv import apply_overrides, piece_mse, valid_lit_overrides
from experiments.lit_si_rule import (RATIO, _to_um, _is_vacv, _unc_vacv_ic,
                                     calibrate_pair_trusted,
                                     calibrate_sentinel_docs, candidate)
from experiments.stacking import fold_arrays
from experiments.stacking_eval import load_sources
from vacv.dataset import load
from vacv.literature import attach
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)

#: типы-отношения из задания + найденный резерв Selectivity Index
EXT_RATIO_TYPES = ["Ratio CC50/EC50", "Ratio EC50", "Ratio IC50", "SI",
                   "Therapeutic index", "Selectivity Index", "Selectivity index"]
TOL_EXACT = 0.003   # бит-в-бит допуск, как валидация Rule A
TOL_AGREE = 0.005   # согласование нескольких записей строки, как candidate()
CC_TYPES = {"CC50", "TD50", "MCC", "TC50", "MTC", "CT50", "LC50", "MNTD", "CD50",
            "GI50", "TGI", "MTD"}
MIN_WITNESS = 3     # как SENTINEL_MIN_ROWS


def ikey(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return MolToInchiKey(m) if m else ""


def build_row_ctx(records: dict, frames: list[pd.DataFrame]) -> dict[str, dict]:
    """y-независимая предобработка: по каждому ключу — кандидат-записи SI-типов,
    наличие VACV-полной пары в любом доке, наличие production-Ratio."""
    ctx: dict[str, dict] = {}
    for frame in frames:
        for smi in frame["SMILES"]:
            if smi in ctx:
                continue
            key = ikey(smi)
            recs = records.get(key, [])
            by_doc: dict[str, list[dict]] = {}
            for r in recs:
                v = _to_um(r["value"], r["units"])
                if v is None:
                    continue
                by_doc.setdefault(r["doc"], []).append({**r, "v": v})
            has_full_pair = any(
                [r for r in rd if str(r.get("type", "")).upper() in CC_TYPES]
                and _unc_vacv_ic(rd) for rd in by_doc.values())
            ratios = []
            for r in recs:
                t = str(r.get("type") or "")
                if t in EXT_RATIO_TYPES and _is_vacv(r):
                    v = _to_um(r["value"], r["units"])
                    if v is not None and v > 0:
                        ratios.append((t, r["doc"], v))
            prod_ratio = any(str(r.get("type") or "") == RATIO and _is_vacv(r)
                             for r in recs)
            ctx[smi] = {"key": key, "ratios": ratios, "has_full_pair": has_full_pair,
                        "prod_ratio": prod_ratio}
    return ctx


def row_ok(row_smi: str, ctx: dict, ptr_fit: set[str], records: dict) -> bool:
    c = ctx[row_smi]
    if not c["ratios"] or c["has_full_pair"] or c["prod_ratio"]:
        return False
    if candidate(records, c["key"], pair_trusted=ptr_fit) is not None:
        return False
    return True


def calibrate_ratio_pairs(frame: pd.DataFrame, y: np.ndarray, ctx: dict,
                          ptr_fit: set[str], records: dict, min_witness: int):
    """Пары (type, doc), валидированные ПО ДАННОЙ fit-выборке (0.3% допуск)."""
    witnesses: dict[tuple[str, str], int] = {}
    contra: set[tuple[str, str]] = set()
    for idx, row in enumerate(frame.itertuples()):
        smi = row.SMILES
        if smi not in ctx or not row_ok(smi, ctx, ptr_fit, records):
            continue
        yy = y[idx]
        for t, d, v in ctx[smi]["ratios"]:
            if abs(v - yy) <= TOL_EXACT * max(v, yy, 1e-9):
                witnesses[(t, d)] = witnesses.get((t, d), 0) + 1
            else:
                contra.add((t, d))
    return {k for k, n in witnesses.items() if n >= min_witness and k not in contra}


def extend_valid(frame_valid: pd.DataFrame, validated: set, ctx: dict,
                 ptr_fit: set[str], records: dict, covered: set[str],
                 prod_keys: set[str]) -> dict[str, float]:
    """Применение кандидат-правила к valid: si-оверрайд для строк без production-хита."""
    out: dict[str, float] = {}
    for row in frame_valid.itertuples():
        smi = row.SMILES
        if smi in covered or smi in prod_keys:
            continue
        if smi not in ctx or not row_ok(smi, ctx, ptr_fit, records):
            continue
        vals = [v for t, d, v in ctx[smi]["ratios"] if (t, d) in validated]
        if not vals:
            continue
        if max(vals) - min(vals) > TOL_AGREE * max(max(vals), 1e-9):
            continue
        out[smi] = float(np.median(vals))
    return out


def run_split(name, src, splits, frame, y, art_idx, covered, records, ctx, rows):
    probs, group = load_sources(src, name)
    pieces = fold_arrays(probs, group, frame, y, splits[name])
    for fold, piece in enumerate(pieces):
        valid = piece["valid"]
        fit_mask = np.ones(len(y), dtype=bool)
        fit_mask[valid] = False
        fit_frame = frame.iloc[fit_mask].reset_index(drop=True)
        valid_frame = frame.iloc[valid]
        sdocs_fit = calibrate_sentinel_docs(records, fit_frame, y[fit_mask])
        ptr_fit = calibrate_pair_trusted(records, fit_frame, y[fit_mask])

        lit_c = valid_lit_overrides(records, valid_frame, sdocs_fit, ptr_fit, covered)
        piece["frame"] = apply_overrides(valid_frame, art_idx, lit_c)
        mse_c = piece_mse(piece, None)

        res = {"split": name, "fold": fold, "mse_prod": mse_c, "n_lit_prod": len(lit_c)}
        y_by_smi = dict(zip(valid_frame["SMILES"], piece["y"]))
        for tag, mw in (("m3", MIN_WITNESS), ("m1", 1)):
            validated = calibrate_ratio_pairs(fit_frame, y[fit_mask], ctx, ptr_fit,
                                              records, mw)
            extra = extend_valid(valid_frame, validated, ctx, ptr_fit, records,
                                 covered, set(lit_c))
            merged = {**lit_c, **{s: {"sentinel": False, "si": v}
                                  for s, v in extra.items() if s not in lit_c}}
            piece["frame"] = apply_overrides(valid_frame, art_idx, merged)
            mse = piece_mse(piece, None)
            exact = wrong = 0
            for s, v in extra.items():
                if abs(v - y_by_smi[s]) <= TOL_EXACT * max(v, y_by_smi[s], 1e-9):
                    exact += 1
                else:
                    wrong += 1
            res[f"mse_{tag}"] = mse
            res[f"pairs_{tag}"] = ";".join(f"{t}@{d}" for t, d in sorted(validated)) or "—"
            res[f"n_{tag}"] = len(extra)
            res[f"exact_{tag}"] = exact
            res[f"wrong_{tag}"] = wrong
        rows.append(res)
        print(f"{name} fold{fold}: prod {mse_c:,.0f} | m3 {res['mse_m3']:,.0f} "
              f"(пар {res['pairs_m3'].count(';') + 1 if res['pairs_m3'] != '—' else 0}, "
              f"оверрайдов {res['n_m3']}, бит {res['exact_m3']}, мимо {res['wrong_m3']}) | "
              f"m1 {res['mse_m1']:,.0f} (оверрайдов {res['n_m1']}, бит {res['exact_m1']}, "
              f"мимо {res['wrong_m1']})", flush=True)


def main() -> None:
    data = load()
    frame, y = attach_nn(attach(data.train)), np.asarray(data.y, dtype=float)
    frame = frame.reset_index(drop=True)
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")

    art_all = pd.concat([pd.read_csv(ROOT / "data/processed/article_si.csv"),
                         pd.read_csv(ROOT / "data/processed/paper_si.csv")],
                        ignore_index=True)
    covered = set(art_all["smiles"])
    art_idx = art_all.set_index("smiles")
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))
    ctx = build_row_ctx(records, [frame, test])

    # -------- статика (y-независимая): инвентарь и покрытие
    inv = []
    for t in EXT_RATIO_TYPES:
        n = vacv_n = 0
        keys = set()
        for recs in records.values():
            for r in recs:
                if str(r.get("type") or "") == t:
                    n += 1
                    if _is_vacv(r):
                        vacv_n += 1
        for smi, c in ctx.items():
            if any(tt == t for tt, _, _ in c["ratios"]):
                keys.add(smi)
        inv.append({"type": t, "записей": n, "VACV": vacv_n,
                    "train+test строк с записью (значимых)": len(keys)})
    inv_df = pd.DataFrame(inv)

    # -------- диагностика: пары на полной train (НЕ источник правила)
    all_ptrs = calibrate_pair_trusted(records, frame, y)
    diag_full = {}
    for mw in (1, MIN_WITNESS):
        v = calibrate_ratio_pairs(frame, y, ctx, all_ptrs, records, mw)
        diag_full[mw] = sorted(v)

    # -------- fold-internal CV
    splits = {f"random-{s}": [v for _, v in KFold(5, shuffle=True, random_state=s).split(y)]
              for s in (0, 1, 2, 3, 4, 5)}
    splits["scaffold"] = scaffold_folds(scaffolds_of(frame), n_splits=5)
    rows: list[dict] = []
    for name in ("random-0", "random-1", "random-2"):
        run_split(name, ROOT / "outputs/stacking-screen", splits, frame, y,
                  art_idx, covered, records, ctx, rows)
    for name in ("random-3", "random-4", "random-5", "scaffold"):
        run_split(name, ROOT / "outputs/stacking-confirm", splits, frame, y,
                  art_idx, covered, records, ctx, rows)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "ratio_types_cv.csv", index=False)

    # -------- отчёт
    def md_table(df: pd.DataFrame) -> str:
        if df is None or not len(df):
            return "_нет_"
        head = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "|" + "|".join("---" for _ in df.columns) + "|"
        body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))
        return "\n".join([head, sep, body])

    lines = [
        "# Задача 6: SI-записи других типов — валидация (type, doc) и расширение candidate()",
        "",
        "Метод: fold-internal (эталон `experiments/lit_si_cv.py`): пары (type, doc)",
        "калибруются только по fit=4/5 (бит-в-бит ≤0.3% + конфликт-правило",
        "lit_si_rule.candidate: полная VACV-пара в любом доке или production-Ratio →",
        "строку не трогаем), применяются к valid=1/5; оценка сохранёнными OOF-прокси.",
        "",
        "## Инвентарь (y-независимо)",
        "",
        md_table(inv_df),
        "",
        "`Ratio IC50` (193) и `Therapeutic index` (14) не имеют ни одной VACV-записи —",
        "расширение candidate() этими типами ничего не добавит. Реальные резервы:",
        "`Ratio CC50/EC50` (21 VACV), `Ratio EC50` (9 VACV), `SI` (5 VACV) и",
        "неиспользуемый `Selectivity Index/index` (30+5 VACV).",
        "",
        "## Диагностические пары на полной train (не источник правила)",
        "",
        f"- без минимума (≥1): {diag_full[1]}",
        f"- c минимумом ≥{MIN_WITNESS}: {diag_full[MIN_WITNESS]}",
        "",
        "## Fold-internal CV (35 фолдов)",
        "",
        md_table(res[["split", "fold", "mse_prod", "mse_m3", "mse_m1", "n_m3",
                      "exact_m3", "wrong_m3", "n_m1", "exact_m1", "wrong_m1"]]),
        "",
        f"- средние: prod {res['mse_prod'].mean():,.0f} | m3 {res['mse_m3'].mean():,.0f} "
        f"({(res['mse_m3'].mean() - res['mse_prod'].mean()) / res['mse_prod'].mean():+.3%}) | "
        f"m1 {res['mse_m1'].mean():,.0f} "
        f"({(res['mse_m1'].mean() - res['mse_prod'].mean()) / res['mse_prod'].mean():+.3%})",
        f"- оверрайдов на valid: m3 Σ{int(res['n_m3'].sum())} (бит-в-бит "
        f"{int(res['exact_m3'].sum())}, мимо {int(res['wrong_m3'].sum())}); m1 "
        f"Σ{int(res['n_m1'].sum())} (бит {int(res['exact_m1'].sum())}, мимо {int(res['wrong_m1'].sum())})",
        f"- пары, которые хоть раз нашли fit-фолды (m3): "
        f"{sorted(set(';'.join(str(x) for x in s.split(';')) for s in res['pairs_m3'] if s != '—'))}",
        "",
        "## Предложение по расширению candidate()",
        "",
        "Расширение легитимно только через калибровку по fit (как PAIR_TRUSTED):",
        "`EXT_RATIO_TYPES = {\"Ratio CC50/EC50\", \"Selectivity Index\", \"Selectivity index\",",
        " \"Ratio EC50\", \"SI\"}` со строго той же схемой: допуск 0.3% бит-в-бит на fit,",
        "конфликт-правило (полная VACV-пара/production-Ratio в других доках → пропуск),",
        "минимум согласованных fit-строк; согласование записей строки 0.5%.",
        "Честная сила правила — в таблице CV выше (оверрайды на valid и их точность).",
    ]
    (OUT / "ratio_types.md").write_text("\n".join(lines), encoding="utf-8")
    print(inv_df.to_string(index=False))
    print("\nдиагн. пары (полная train): m1", diag_full[1], "| m3", diag_full[MIN_WITNESS])
    print("\nсредние:", f"prod {res['mse_prod'].mean():,.0f}",
          f"m3 {res['mse_m3'].mean():,.0f}", f"m1 {res['mse_m1'].mean():,.0f}")
    print(f"оверрайды на valid: m3 Σ{int(res['n_m3'].sum())} бит {int(res['exact_m3'].sum())} "
          f"мимо {int(res['wrong_m3'].sum())} | m1 Σ{int(res['n_m1'].sum())} "
          f"бит {int(res['exact_m1'].sum())} мимо {int(res['wrong_m1'].sum())}")
    print(f"→ {OUT / 'ratio_types.md'}, {OUT / 'ratio_types_cv.csv'}")


if __name__ == "__main__":
    main()
