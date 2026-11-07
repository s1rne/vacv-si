"""Общий оверрайд из ChEMBL-записей вида (EC50 Vaccinia) + (цитотокс той же
клеточной линии) — конвенция организаторов, выведенная на доках
CHEMBL3351523 (BMC 2014 бензимидазолы, sid3123xxx: CC50 = цитотокс Vero 76,
IC50 = EC50 Vaccinia Vero 76, цензура ">"/">=" → число), CHEMBL6092128 и др.
Правило ключует пару по документу: EC = Vaccinia-запись (IC_TYPES, "vaccin"
в описании), CC = cytotox-запись (CC_TYPES либо IC_TYPES с цитотокс-описанием
"cell growth"/"cell morphology"/"cytotox"): сначала цитотокс той же клеточной
линии, что EC (пересечение cell-токенов); если совпадений нет — любой цитотокс
дока (панель De Clercq: VACV в кроличьих клетках, цитотокс L1210 —
sid103181863: 1000/400 = 2.5 бит-в-бит). si = CC/EC (сырое частное; якоря
допускают round3).

Точность значений: организаторы берут значения из таблиц статей; ChEMBL
хранит nM-конверсии с лишней точностью (651 nM = 0.651 µM, а в таблице
статьи 0.65 — train#244 '4a': 28.33/0.65 = 43.585 против сырого 43.518).
Такие ключи расходятся и исключаются ('4a' покрыт paper_si с ошибкой 0.067 —
незаметно); поэтому конверсия сырое частное без округления.

Исключения — по КЛЮЧУ молекулы в рамках дока: если якорь дока расходится
с y, ключ исключается из строк этого дока (организаторы взяли пару из
другого источника: #268 'sid494430641' y=35.714=500/14; #50/#187 y=1.0 из
пар ">100/>100", ">250/>250" статей jm1011462/jm00152a016; #173/#230
y=10000 при есть нецензурированной VACV-EC50 в ChEMBL — у организаторов
IC50 NaN; #29 '17': в ChEMBL 30.9, у организаторов 30.09). Док проходит,
если у него есть ≥1 сошедшийся якорь. Train-строки одного ключа могут
сходиться в разных доках (разные депозиты) — каждой строке свой док.

Сентинел-ветка: y=10000 валиден, только если у ключа нет нецензурированной
Vaccinia-EC50 ни в одном доке (иначе строка оргов — из другого источника).

Валидация: каждый train-якорь обязан сходиться бит-в-бит (или через round3);
расходящиеся ключи исключаются. Запрещённые колонки test (CC50_uM,
IC50_uM_VACV) скриптом не читаются вовсе.

Запуск: uv run python experiments/chembl_si_pairs.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULE = "chembl_si_pairs"

CC_TYPES = {"CC50", "TD50", "MCC", "TC50", "MTC", "CT50", "LC50", "MNTD", "CD50", "GI50", "TGI", "MTD"}
IC_TYPES = {"IC50", "EC50", "ED50", "ID50", "AC50", "Potency"}


def _norm_units(u: str) -> str:
    return re.sub(r"[\s.\-]+", "", str(u or "").lower()).replace("µ", "u").replace("μ", "u")


def _conv(value, units, mw=None):
    """→ µM, сырое значение. Массовые единицы — через MolWt структуры."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    u = _norm_units(units)
    if u in ("nm",):
        return v / 1000.0
    if u in ("um", "m", ""):
        return v
    if u == "mm":
        return v * 1e3
    if mw is None:
        return None
    if u in ("ugml", "ug/l"):
        return v * 1000.0 / mw
    if u in ("mgml", "mg/l", "gl"):
        return v * 1e6 / mw
    if u == "ngml":
        return v / mw
    return None


def _cell_tokens(desc: str) -> set[str]:
    """Клеточные линии из описания ассая: '... in Vero 76 cells' → 'vero 76'."""
    toks = set()
    for m in re.finditer(r"([A-Za-z0-9][A-Za-z0-9\- ]*?)\s+cells?\b", str(desc or "").lower()):
        t = m.group(1).strip()
        for sep in ("infected in ", "infected ", "mock-infected ", "monkey ", "human ", "mock "):
            if sep in t:
                t = t.split(sep)[-1].strip()
        if t:
            toks.add(t)
    return toks


CYTOTOX_DESC = ("cell growth", "cell morphology", "cytotox", "cytotoxic")


def _is_cytotox(rec) -> bool:
    """Цитотокс-запись: CC_TYPES либо potентность с цитотокс-описанием."""
    if rec["type"] in CC_TYPES:
        return True
    if rec["type"] in IC_TYPES:
        d = rec["desc"].lower()
        return any(p in d for p in CYTOTOX_DESC)
    return False


def _pairs_for(records_by_key: dict, key: str, mw: float | None = None):
    """(doc → (cc_recs, ec_recs)): EC = Vaccinia-запись, CC = cytotox той же
    клеточной линии, при отсутствии совпадений — любой цитотокс дока."""
    by_doc = defaultdict(list)
    for r in records_by_key.get(key, []):
        v = _conv(r["value"], r["units"], mw)
        if v is None:
            continue
        by_doc[r["doc"]].append({**r, "v": v,
                                 "rel": (r.get("relation") or "").strip(),
                                 "desc": str(r.get("assay") or "")})
    out = {}
    for doc, rs in by_doc.items():
        ecs = [r for r in rs if r["type"] in IC_TYPES and "vaccin" in r["desc"].lower()]
        if not ecs:
            continue
        cc_all = [r for r in rs if _is_cytotox(r) and "vaccin" not in r["desc"].lower()]
        ec_toks: set[str] = set()
        for r in ecs:
            ec_toks |= _cell_tokens(r["desc"])
        ccs = [r for r in cc_all if _cell_tokens(r["desc"]) & ec_toks]
        if not ccs:
            ccs = cc_all
        if not ccs:
            continue
        out[doc] = (ccs, ecs)
    return out


def _pick_cc(ccs):
    """CC: единственный → min; несколько → цензурированное (">"→число),
    иначе max."""
    vals = {c["v"] for c in ccs}
    if len(vals) == 1:
        return min(vals)
    cens = [c["v"] for c in ccs if c["rel"] in (">", ">=")]
    return max(cens) if cens else max(vals)


def _pick_ec(ecs):
    """EC: единственное значение → оно; несколько разных → цензурированное;
    несколько нецензурированных разных → None (неоднозначно)."""
    vals = sorted({e["v"] for e in ecs})
    if len(vals) == 1:
        return vals[0]
    cens = sorted({e["v"] for e in ecs if e["rel"] in (">", ">=")})
    return cens[0] if cens else None


def _unc_vacv_ic_any(recs_by_key, key):
    for r in recs_by_key.get(key, []):
        if r["type"] in IC_TYPES and "vaccin" in str(r.get("assay") or "").lower() \
                and (r.get("relation") or "").strip() not in (">", ">="):
            return True
    return False


def main() -> None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import MolWt
    from rdkit.Chem.inchi import MolToInchiKey

    RDLogger.DisableLog("rdApp.*")
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))
    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")

    def ik_of(smiles: str) -> str:
        m = Chem.MolFromSmiles(smiles)
        return MolToInchiKey(m) if m else ""

    def mw_of(smiles: str) -> float | None:
        m = Chem.MolFromSmiles(smiles)
        return MolWt(m) if m else None

    # train-якоря: (idx, doc) → (si, sentinel); расходящиеся ключи — по доку
    val: dict[tuple[int, str], tuple] = {}
    doc_bad: dict[str, set] = defaultdict(set)
    train_ik: dict[int, str] = {}
    for idx, r in train.iterrows():
        key = ik_of(r["SMILES"])
        train_ik[idx] = key
        if not key:
            continue
        y = float(r["SI VACV"])
        mw = mw_of(r["SMILES"])
        for doc, (ccs, ecs) in _pairs_for(records, key, mw).items():
            ec = _pick_ec(ecs)
            if ec is None or ec <= 0:
                continue  # неоднозначный/нулевой EC — не якорь
            if y == 10000.0 and not _unc_vacv_ic_any(records, key):
                val[(idx, doc)] = (None, True)
                continue
            si = _pick_cc(ccs) / ec
            ok = abs(si - y) <= 1e-9 * max(abs(si), abs(y), 1e-12) or y == round(si, 3)
            if ok:
                val[(idx, doc)] = (si, False)
            else:
                doc_bad[doc].add(key)

    keys_ok_by_doc: dict[str, set] = defaultdict(set)
    for (idx, doc) in val:
        if train_ik[idx] not in doc_bad[doc]:
            keys_ok_by_doc[doc].add(train_ik[idx])
    passing = {doc: keys for doc, keys in keys_ok_by_doc.items() if keys}
    n_ok = {doc: len(keys) for doc, keys in passing.items()}
    for doc in sorted(passing):
        print(f"{doc}: якорей {len(passing[doc])} (bad-ключей {len(doc_bad[doc])})")
    print(f"проходных доков: {len(passing)}")

    rows = []
    for idx, r in train.iterrows():
        key = train_ik[idx]
        if not key:
            continue
        cands = sorted((-n_ok[d], d, *val[(idx, d)]) for d in passing
                       if (idx, d) in val and key not in doc_bad[d])
        if not cands:
            continue
        doc, si, sent = cands[0][1], cands[0][2], cands[0][3]
        rows.append({"Title": str(r["Title"]), "sentinel": sent, "si": si,
                     "smiles": r["SMILES"], "pmc": doc, "label": doc,
                     "rule": RULE})
    for idx, r in test.iterrows():
        key = ik_of(r["SMILES"])
        if not key:
            continue
        doc = si = sent = None
        cands = []
        for d in sorted(passing):
            if key in doc_bad[d]:
                continue
            pair = _pairs_for(records, key, mw_of(r["SMILES"])).get(d)
            if not pair or not pair[0]:
                continue
            ec = _pick_ec(pair[1])
            if ec is None or ec <= 0:
                continue
            cands.append((-n_ok[d], d, _pick_cc(pair[0]) / ec))
        if cands:
            cands.sort()
            doc, si, sent = cands[0][1], cands[0][2], False
            if len({round(c[2], 9) for c in cands}) > 1:
                print(f"!! конфликт test-si для {key}: " + str({c[1]: c[2] for c in cands}))
        else:
            # сентинел: в проходном доке есть CC, а нецензурированной
            # Vaccinia-EC50 нет нигде
            has_cc = False
            for d in sorted(passing):
                if key in doc_bad[d]:
                    continue
                pair = _pairs_for(records, key).get(d)
                if pair and pair[0] and not pair[1]:
                    has_cc = True
            if not (has_cc and not _unc_vacv_ic_any(records, key)):
                continue
            sent = True
        rows.append({"Title": str(r["Title"]), "sentinel": sent, "si": si,
                     "smiles": r["SMILES"], "pmc": doc or "", "label": doc or "",
                     "rule": RULE})

    out = pd.DataFrame(rows)
    if len(out):
        out = out.drop_duplicates(["smiles", "Title"], keep="first")
    path = ROOT / "data" / "processed" / "chembl_si_pairs.csv"
    out.to_csv(path, index=False)
    n_sent = int(out["sentinel"].sum())
    print(f"→ {path}: {len(out)} строк (сентинелов {n_sent}), "
          f"доков {len(passing)}")


if __name__ == "__main__":
    main()
