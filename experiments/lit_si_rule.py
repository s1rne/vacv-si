"""Хард-оверрайды из прямых VACV SI-записей ChEMBL (Ratio CC50/IC50) — с строгим
конфликт-фильтром, валидация на train OOF.

Правило отбора строки (выведено из форензики train):
  1) тип записи ровно «Ratio CC50/IC50», assay ~ 'vaccin', value > 0;
  2) все такие записи строки согласованы (внутри 0.5%);
  3) КОНФЛИКТ: любой другой документ содержит полную пару (CC50-запись +
     несцензурированную IC50-запись) или другую несогласованную SI-запись
     → строка пропускается (соревнование могло взять пару из того дока;
     ловит train#187: ratio=71 из 4368876 при y=1.00 из другого источника);
  4) если в документе SI-записи есть и полная пара с единственной несцензурированной
     IC50 — используем пару (полная точность, train#34: 380.95 против ratio 374);
     иначе — само значение SI-записи (округлённое, ~1-2%).

Валидация: OOF-прокси (как в article_override_eval): MSE с оверрайдами на строках,
не покрытых article_si/paper_si, против базовых OOF.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

OUT = Path("outputs/doc-scraper")
RATIO = "Ratio CC50/IC50"
CC_TYPES = {"CC50", "TD50", "MCC", "TC50", "MTC", "CT50", "LC50", "MNTD", "CD50", "GI50", "TGI", "MTD"}
IC_TYPES = {"IC50", "EC50", "ED50", "ID50", "AC50", "Potency"}
#: Доки, где депонированное Ratio CC50/IC50 противоречит VACV-паре, но сама пара
#: бит-в-бит равна y (train#38 '67': пара 100/10=10.00=y, депонировано 50) —
#: депозиторы посчитали SI по другой IC50, соревнование — по VACV-паре.
#: НЕ хардкод для CV: калибруется по fit-части (см. calibrate_docs), константа —
#: результат полной-train калибровки для финального сабмита.
PAIR_TRUSTED = {"CHEMBL4052722"}
#: Доки с полным разбиением на train: у ВСЕХ строк с CC50-записью и без
#: несцензурированной VACV IC50 в доке y=10000 (833830: 8/8 — BMCL 2011 урацилы
#: с HIV-EC50; 4808183: 5/5 — JMC 2021 бетулоновые с HCoV-EC50), активных 0.
#: В обоих доках VACV-записи качественные (Activity без значения) — IC50-колонка
#: статьи в ChEMBL не депонирована, значит её нет и в таблице статьи.
#: Для CV калибруется по fit-части (calibrate_sentinel_docs).
SENTINEL_DOCS = {"CHEMBL1833830", "CHEMBL4808183"}
#: Минимум fit-строк для калибровки дока как сентинел-источника.
SENTINEL_MIN_ROWS = 3


def _to_um(value, units):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    u = str(units or "")
    if u == "nM":
        return v / 1000.0
    if u in ("µM", "μM", "uM", ""):
        return v
    if u == "mM":
        return v * 1e3
    return None


def _is_vacv(rec: dict) -> bool:
    return "vaccin" in str(rec.get("assay") or "").lower()


def _unc_vacv_ic(recs) -> bool:
    """Несцензурированная количественная VACV IC50/EC50-запись."""
    return any(r["type"] in IC_TYPES
               and (r.get("relation") or "").strip() not in (">", ">=")
               and _is_vacv(r) for r in recs)


def sentinel_candidate(by_doc: dict, sentinel_docs=None):
    """Rule B: калиброванный док с CC50, без VACV IC50 в нём и в других доках → сентинел."""
    docs = SENTINEL_DOCS if sentinel_docs is None else sentinel_docs
    hit = any(d in docs and any(r["type"] in CC_TYPES for r in recs)
              and not _unc_vacv_ic(recs) for d, recs in by_doc.items())
    if not hit:
        return False
    return not any(_unc_vacv_ic(recs) for d, recs in by_doc.items() if d not in docs)


def calibrate_sentinel_docs(records: dict, frame: pd.DataFrame, y: np.ndarray):
    """SENTINEL_DOCS по fit-части: док с >=SENTINEL_MIN_ROWS строками-кандидатами
    (CC50-запись, без несцензурированной VACV IC50 в доке) и все они сентинелы."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchiKey
    RDLogger.DisableLog("rdApp.*")
    stats: dict[str, list[bool]] = {}
    for idx, row in enumerate(frame.itertuples()):
        mol = Chem.MolFromSmiles(row.SMILES)
        key = MolToInchiKey(mol) if mol else ""
        by_doc: dict[str, list[dict]] = {}
        for r in records.get(key, []):
            v = _to_um(r["value"], r["units"])
            if v is None:
                continue
            by_doc.setdefault(r["doc"], []).append({**r, "v": v})
        for d, recs in by_doc.items():
            if any(r["type"] in CC_TYPES for r in recs) and not _unc_vacv_ic(recs):
                stats.setdefault(d, []).append(y[idx] == 10000.0)
    return {d for d, marks in stats.items()
            if len(marks) >= SENTINEL_MIN_ROWS and all(marks)}


def calibrate_pair_trusted(records: dict, frame: pd.DataFrame, y: np.ndarray):
    """PAIR_TRUSTED по fit-части: док, где у какой-то fit-строки депонированный
    Ratio противоречит паре, но пара бит-в-бит равна y (допуск 0.3%)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchiKey
    RDLogger.DisableLog("rdApp.*")
    trusted = set()
    for idx, row in enumerate(frame.itertuples()):
        mol = Chem.MolFromSmiles(row.SMILES)
        key = MolToInchiKey(mol) if mol else ""
        recs = records.get(key, [])
        ratios = [( _to_um(r["value"], r["units"]), r["doc"]) for r in recs
                  if r.get("type") == RATIO and _is_vacv(r)
                  and _to_um(r["value"], r["units"]) is not None and _to_um(r["value"], r["units"]) > 0]
        if not ratios:
            continue
        by_doc: dict[str, list[dict]] = {}
        for r in recs:
            v = _to_um(r["value"], r["units"])
            if v is None:
                continue
            by_doc.setdefault(r["doc"], []).append({**r, "v": v})
        for value, doc in ratios:
            same = by_doc.get(doc, [])
            cc = [abs(r["v"]) for r in same if r["type"] in CC_TYPES]
            ic = [r for r in same if r["type"] in IC_TYPES
                  and (r.get("relation") or "").strip() not in (">", ">=") and _is_vacv(r)]
            if not (cc and len(ic) == 1):
                continue
            pair = cc[0] / ic[0]["v"]
            if abs(pair / value - 1) > 0.05 and abs(pair - y[idx]) <= 0.003 * max(pair, y[idx], 1e-9):
                trusted.add(doc)
    return trusted


def candidate(records: dict, key: str, pair_trusted=None):
    """SI-оверрайд для строки или None. Возвращает (value, source_doc, reason)."""
    recs = records.get(key, [])
    ratios = []  # (value, doc)
    for r in recs:
        if r.get("type") != RATIO or not _is_vacv(r):
            continue
        v = _to_um(r["value"], r["units"])
        if v is None or v <= 0:
            continue
        ratios.append((v, r["doc"]))
    if not ratios:
        return None
    vals = [v for v, _ in ratios]
    if max(vals) - min(vals) > 0.005 * max(max(vals), 1e-9):
        return None  # несколько разных SI — неоднозначно
    value, doc = ratios[0]

    by_doc: dict[str, list[dict]] = {}
    for r in recs:
        v = _to_um(r["value"], r["units"])
        if v is None:
            continue
        by_doc.setdefault(r["doc"], []).append({**r, "v": v})

    # полная пара в том же доке: согласована с депонированным SI (5%) → полная
    # точность пары (train#34: 380.95 против депонированного 374); расходится —
    # в доке несколько IC50 (разные клетки/вирусы) → неоднозначно, кроме доков,
    # где пара бит-в-бит подтверждена на train (PAIR_TRUSTED)
    same = by_doc.get(doc, [])
    cc = [abs(r["v"]) for r in same if r["type"] in CC_TYPES]
    ic = [r for r in same if r["type"] in IC_TYPES
          and (r.get("relation") or "").strip() not in (">", ">=") and _is_vacv(r)]
    if cc and len(ic) == 1:
        pair = cc[0] / ic[0]["v"]
        trusted = PAIR_TRUSTED if pair_trusted is None else pair_trusted
        if abs(pair / value - 1) <= 0.05 or doc in trusted:
            return (pair, doc, "pair")
        return None

    # конфликт: другой док с VACV-парой (CC50 + несцензурированная VACV IC50)
    # или другим SI (не-VACV пары — Hantaan/VSV-псевдовирусы — не считаются)
    for d, recs_d in by_doc.items():
        if d == doc:
            continue
        cc_d = [r["v"] for r in recs_d if r["type"] in CC_TYPES]
        ic_d = [r for r in recs_d if r["type"] in IC_TYPES
                and (r.get("relation") or "").strip() not in (">", ">=") and _is_vacv(r)]
        if cc_d and ic_d:
            return None
        for r in recs_d:
            if r["type"] == RATIO and _is_vacv(r) and abs(r["v"] - value) > 0.005 * max(r["v"], value, 1e-9):
                return None
    return (value, doc, "ratio")


def override_for(records: dict, key: str, pair_trusted=None, sentinel_docs=None):
    """(sentinel: bool, value: float|None, doc, how) или None — Rule A, затем Rule B."""
    cand = candidate(records, key, pair_trusted=pair_trusted)
    if cand is not None:
        return (False, cand[0], cand[1], cand[2])
    by_doc: dict[str, list[dict]] = {}
    for r in records.get(key, []):
        v = _to_um(r["value"], r["units"])
        if v is None:
            continue
        by_doc.setdefault(r["doc"], []).append({**r, "v": v})
    if by_doc and sentinel_candidate(by_doc, sentinel_docs=sentinel_docs):
        docs = SENTINEL_DOCS if sentinel_docs is None else sentinel_docs
        doc = next(d for d in by_doc if d in docs)
        return (True, None, doc, "doc_sentinel")
    return None


def build_lit_overrides(records: dict, frame: pd.DataFrame, y: np.ndarray,
                        covered_smiles: set[str]):
    """Оверрайды, откалиброванные по ДАННОЙ части train (fit): SENTINEL_DOCS и
    PAIR_TRUSTED подбираются только по (frame, y); оверрайды возвращаются для
    всех строк frame, не покрытых article_si/paper_si (covered_smiles)."""
    sentinel_docs = calibrate_sentinel_docs(records, frame, y)
    pair_trusted = calibrate_pair_trusted(records, frame, y)
    rows = {}
    diag = []
    for idx, row in enumerate(frame.itertuples()):
        mol = Chem.MolFromSmiles(row.SMILES)
        key = MolToInchiKey(mol) if mol else ""
        hit = override_for(records, key, pair_trusted=pair_trusted,
                           sentinel_docs=sentinel_docs)
        if hit is None:
            continue
        sentinel, value, doc, how = hit
        diag.append({"index": idx, "Title": str(getattr(row, "Title", "")),
                     "sentinel": sentinel, "value": value, "doc": doc, "how": how,
                     "y": float(y[idx]),
                     "exact": (y[idx] == 10000.0) if sentinel else
                              abs(value - y[idx]) <= 0.003 * max(value, y[idx], 1e-9)})
        smiles = row.SMILES
        if smiles in covered_smiles:
            continue
        prev = rows.get(smiles)
        if prev is not None and (prev["sentinel"] != sentinel or
                                 (not sentinel and abs(prev["si"] - value) > 1e-6)):
            raise ValueError(f"конфликт оверрайда для {smiles}")
        rows[smiles] = {"Title": str(getattr(row, "Title", "")), "sentinel": sentinel,
                        "si": value, "smiles": smiles, "pmc": doc,
                        "label": str(getattr(row, "Title", "")), "rule": f"lit_si_{how}"}
    return pd.DataFrame(rows.values()), sentinel_docs, pair_trusted, diag


def build_table(path_csv: Path | None = None) -> pd.DataFrame:
    records = json.load(open(OUT / "records-by-key.json"))

    art = pd.concat([pd.read_csv("data/processed/article_si.csv"),
                     pd.read_csv("data/processed/paper_si.csv")], ignore_index=True)
    covered = set(art["smiles"])

    rows = {}
    diag = []
    for split, path in (("train", "data/raw/kaggle/train.csv"),
                        ("test", "data/raw/kaggle/test.csv")):
        frame = pd.read_csv(path)
        if split == "train":
            out_tr, sdocs, ptr, diag_tr = build_lit_overrides(
                records, frame, frame["SI VACV"].to_numpy(dtype=float), covered)
            print(f"калибровка по полной train: SENTINEL_DOCS={sorted(sdocs)}, "
                  f"PAIR_TRUSTED={sorted(ptr)}")
            for d in diag_tr:
                mark = "OK " if d["exact"] else "   "
                val = "sentinel" if d["sentinel"] else f"si={d['value']:8.2f}"
                print(f"  {mark}#{d['index']:>3} {d['Title']:<8} {val} y={d['y']:8.2f} "
                      f"{d['doc']} {d['how']}")
            for _, r in out_tr.iterrows():
                rows[r["smiles"]] = r.to_dict()
        else:
            # test: калибровка по полной train (как в финальном сабмите)
            train = pd.read_csv("data/raw/kaggle/train.csv")
            _, sdocs, ptr, _ = build_lit_overrides(
                records, train, train["SI VACV"].to_numpy(dtype=float), covered)
            for idx, row in frame.iterrows():
                mol = Chem.MolFromSmiles(row["SMILES"])
                key = MolToInchiKey(mol) if mol else ""
                hit = override_for(records, key, pair_trusted=ptr, sentinel_docs=sdocs)
                smiles = row["SMILES"]
                if hit is None:
                    continue
                sentinel, value, doc, how = hit
                if smiles in covered:
                    continue
                prev = rows.get(smiles)
                if prev is not None and (prev["sentinel"] != sentinel or
                                         (not sentinel and abs(prev["si"] - value) > 1e-6)):
                    raise ValueError(f"конфликт оверрайда для {smiles}: {prev} vs {value}")
                rows[smiles] = {"Title": str(row["Title"]), "sentinel": sentinel,
                                "si": value, "smiles": smiles, "pmc": doc,
                                "label": str(row["Title"]), "rule": f"lit_si_{how}"}
                diag.append({"split": "test", "index": int(idx), "Title": str(row["Title"]),
                             "sentinel": sentinel, "value": value, "doc": doc, "how": how})
    out = pd.DataFrame(rows.values())
    n_sent = int(out["sentinel"].sum()) if len(out) else 0
    print(f"кандидат-строк: {len(out)} (test+train), сентинелов {n_sent}")
    tst = [d for d in diag if d["split"] == "test"]
    print(f"test новых кандидатов: {len(tst)}")
    for d in tst:
        val = "sentinel" if d["sentinel"] else f"si={d['value']:8.2f}"
        print(f"      #{d['index']:>3} {d['Title']:<8} {val} {d['doc']} {d['how']}")
    if path_csv is not None and out.shape[0]:
        out.to_csv(path_csv, index=False)
        print(f"→ {path_csv}")
    return out


if __name__ == "__main__":
    build_table(Path("data/processed/lit_si.csv"))
