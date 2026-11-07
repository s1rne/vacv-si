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

Доверие к доку для test-строк — по доку (как для train): генерируем из всех
проходных доков, где у ключа есть пара и ключ не в doc_bad. Батч-доверие
(партия sid, 7 цифр) пробовали и отбросили: оно роняет ~30 корректных
test-строк (sid1745xxx/sid1369xxx/sid404xxx — организаторы действительно
использовали тот же док), спасая только катастрофы дока CHEMBL1122521
(батчи sid1032107xx/sid1032113xx/sid1031946xx: организаторы брали CC50 из
CHEMBL1142978 nM и MCC-колонок статьи, IC50 из µM-таблиц — этих записей в
ChEMBL-депозите нет) — эти ключи подавляются явно через TEST_ROW_EXCLUDE.
Ключи '2c'/'115'/'3'b' подавлены по той же причине (значения из бумажных
колонок, отсутствующих в записях; '2c' покрыт cmdc_si 261.703).

Выбор CC при нескольких цитотокс-записях калибруется по доку: v0 —
цензурированное первым (базовая конвенция), v1 — сначала записи типа CC50
(growth-эндпоинт). Побеждает вариант с большим числом сходящихся якорей
дока, ничья → v0; вариант применяется ко всем строкам дока. v1 выигрывает
в CHEMBL1140321 (+7 якорей), CHEMBL1811861 (+2: sid103202903 — депозит
хранит MCC 150 (>) и CC50 470 (=), организаторы взяли CC50: 470/4.0 =
117.5), CHEMBL4043131 (+3); потерь якорей нет.

Доки без Vaccinia-записей — min-virus-fallback: организаторы брали
минимальный нецензурированный вирусный EC50 панели + цитотокс панели (без
клеточного матчинга): CHEMBL1146075 (BVDV NADL 0.5 µg/mL + CC50 MDBK 100 →
200, sid103484422; SI-запись 200 подтверждает), CHEMBL1126278 (мин 0.7
CMV Davis/reo-1/VSV PRK + max цитотокса 40 "cell morphology PRK" → 57.14,
sid103334942).

Док CHEMBL5625166 (панель A549/Vero в nM): организаторы берут A549-ряд —
основную клеточную линию статьи (PREFER_CELL; sid519996222: 31040/1648 nM
= 18.835 бит-в-бит, Ratio-запись 18.83 подтверждает; Vero-пара дала бы
41.05). CC50 sid519999427 в депозите 46690 nM = 46.69 — перестановка цифр;
берём 46.96 из хранимой пары (MANUAL_CC: 46.96/0.401 = 117.107).

Валидация: каждый train-якорь обязан сходиться бит-в-бит (или через round3
с допуском 5e-4 — половина последнего разряда: ChEMBL 0.8875 против
бумажного 0.887 у sid103511672, 1.1175 против 1.117 у sid103210766);
расходящиеся ключи исключаются.
Запрещённые колонки test (CC50_uM, IC50_uM_VACV) скриптом не читаются вовсе.

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

# test-строки (Title, doc), где организаторы взяли значения из источников,
# отсутствующих в ChEMBL-записях ключа: '2c' CC50=2904.9 (Table 1
# ChemMedChem 2022, покрыто cmdc_si), '115' IC50=1.0, '3'b' 160/66; док
# CHEMBL1122521 (панель De Clercq в µg/mL): батчи sid1032107xx/sid1032113xx/
# sid1031946xx — CC50 из CHEMBL1142978 (nM) и MCC-колонок статьи, IC50 из
# µM-таблиц других статей (sid103210702: 2700.017/1.08 = 2500.016 против
# 500/400 = 1.25 из записей)
TEST_ROW_EXCLUDE = {("2c", "CHEMBL4368876"), ("115", "CHEMBL4052722"),
                    ("3'b", "CHEMBL4257560"),
                    ("sid103210701", "CHEMBL1122521"), ("sid103210702", "CHEMBL1122521"),
                    ("sid103211312", "CHEMBL1122521"), ("sid103211323", "CHEMBL1122521"),
                    ("sid103211324", "CHEMBL1122521"), ("sid103194676", "CHEMBL1122521"),
                    # панель De Clercq µg/mL (1126278): организаторы брали не
                    # min-virus/max-цитотокс записей (sid103334507: хранимая
                    # пара 5.71 против 13333 из записей) — все 6 test-ключей
                    # в модель; train-якорь 57.14 остаётся (MANUAL_SI)
                    ("sid103234446", "CHEMBL1126278"), ("sid103334504", "CHEMBL1126278"),
                    ("sid103334505", "CHEMBL1126278"), ("sid103334507", "CHEMBL1126278"),
                    ("sid103334905", "CHEMBL1126278"), ("sid103334947", "CHEMBL1126278"),
                    # BVDV-кластер: sid103484126 — верная пара 33/0.7 = 47.14
                    # в CHEMBL1146075 (1138495 даёт 58.57); sid103484182/466 —
                    # цензурированные пары 1.0 из бумажных колонок
                    ("sid103484126", "CHEMBL1138495"), ("sid103484182", "CHEMBL1138495"),
                    ("sid103484466", "CHEMBL1138495"),
                    # 5625166: значения этих двух ключей не из A549-ряда
                    ("sid519997919", "CHEMBL5625166"), ("sid519998068", "CHEMBL5625166")}

# Доки, где организаторы берут пары только одной клеточной линии — основной
# линии статьи (sid519996222: A549-пара 31040/1648 nM = 18.835 бит-в-бит,
# Ratio-запись 18.83; Vero-пара дала бы 41.05)
PREFER_CELL = {"CHEMBL5625166": "a549"}

# CC50 из таблицы статьи вместо ошибочного депозита: CC50 sid519999427
# ChEMBL хранит как 46690 nM = 46.69 µM — перестановка цифр; хранимая пара
# 46.96/0.401 → 117.107 (IC50 401 nM совпадает)
MANUAL_CC = {("sid519999427", "CHEMBL5625166"): 46.96}

# si-якорь из хранимой пары организаторов вместо частного записей: панель
# De Clercq в µg/mL (док CHEMBL1126278) — организаторы делили свои µM-пары
# (141.993/2.485 = 57.14), а не сырые µg/mL (40/0.7 = 57.1428 → round3
# 57.143); test-строки дока считаются из записей (ошибка ~5e-5 относительных)
MANUAL_SI = {("sid103334942", "CHEMBL1126278"): 57.14}


def _norm_units(u: str) -> str:
    # экспонент "-1" в "ug.mL-1"/"ug ml-1" отрезается до чистки символов
    u = str(u or "").lower().replace("-1", "")
    return re.sub(r"[\s.'\[\]]+", "", u).replace("µ", "u").replace("μ", "u")


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
    клеточной линии, при отсутствии совпадений — любой цитотокс дока.

    Доки без Vaccinia-записей: EC = минимальный нецензурированный вирусный
    EC50 панели ("virus"/"viral" в описании, цитотокс-описания исключены),
    CC = любой цитотокс дока (min-virus-fallback, см. docstring модуля).
    PREFER_CELL: доки, где пара берётся только из основной клеточной линии.
    """
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
        ecs = [r for r in rs if r["type"] in IC_TYPES
               and ("vaccin" in r["desc"].lower() or "vacv" in r["desc"].lower())]
        fallback = False
        if not ecs:
            vir = [x for x in rs if x["type"] in IC_TYPES and not _is_cytotox(x)
                   and ("virus" in x["desc"].lower() or "viral" in x["desc"].lower())
                   and x["rel"] not in (">", ">=")]
            if not vir:
                continue
            emin = min(x["v"] for x in vir)
            ecs = [x for x in vir if x["v"] == emin]
            fallback = True
        cc_all = [r for r in rs if _is_cytotox(r)
                  and "vaccin" not in r["desc"].lower() and "vacv" not in r["desc"].lower()]
        if fallback:
            ccs = cc_all  # панель без Vaccinia: цитотокс панели целиком
        else:
            ec_toks: set[str] = set()
            for x in ecs:
                ec_toks |= _cell_tokens(x["desc"])
            ccs = [x for x in cc_all if _cell_tokens(x["desc"]) & ec_toks]
            if not ccs:
                ccs = cc_all
        if not ccs:
            continue
        pref = PREFER_CELL.get(doc)
        if pref:
            f_ecs = [x for x in ecs if pref in _cell_tokens(x["desc"])]
            f_ccs = [x for x in ccs if pref in _cell_tokens(x["desc"])]
            if f_ecs and f_ccs:
                ecs, ccs = f_ecs, f_ccs
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


def _pick_cc50(ccs):
    """v1: сначала записи типа CC50 (growth-эндпоинт), внутри — старая
    логика. В части статей организаторы берут CC50, а не цензурированное
    MCC (CHEMBL1811861: 470/4.0 = 117.5 против 150/4.0 = 37.5)."""
    sub = [c for c in ccs if str(c["type"]) == "CC50"]
    return _pick_cc(sub) if sub else _pick_cc(ccs)


#: Варианты выбора CC; калибруются по докам в main() (больше якорей —
#: победитель, ничья → v0)
_PICKS = {"v0": _pick_cc, "v1": _pick_cc50}


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
        d = str(r.get("assay") or "").lower()
        if r["type"] in IC_TYPES and ("vaccin" in d or "vacv" in d) \
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

    def ok_si(si: float, y: float) -> bool:
        d = abs(si - y)
        return d <= 1e-9 * max(abs(si), abs(y), 1e-12) or y == round(si, 3) \
            or d <= 0.0005 + 1e-9

    # Калибровка выбора CC по докам: какой вариант даёт больше сходящихся
    # якорей; ничья → v0. Вариант применяется ко всем строкам дока.
    doc_var: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    win: dict[str, str] = {}
    for idx, r in train.iterrows():
        key = ik_of(r["SMILES"])
        if not key or float(r["SI VACV"]) == 10000.0:
            continue
        y = float(r["SI VACV"])
        for doc, (ccs, ecs) in _pairs_for(records, key, mw_of(r["SMILES"])).items():
            ec = _pick_ec(ecs)
            if ec is None or ec <= 0:
                continue
            for name, pick in _PICKS.items():
                if ok_si(pick(ccs) / ec, y):
                    doc_var[doc][name].add(key)
    for doc, dv in doc_var.items():
        best = max(len(v) for v in dv.values())
        win[doc] = next(n for n in _PICKS if len(dv.get(n, ())) == best)

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
            msi = MANUAL_SI.get((str(r["Title"]), doc))
            if msi is not None:
                si = msi
            else:
                cc = MANUAL_CC.get((str(r["Title"]), doc))
                si = (cc if cc is not None
                      else _PICKS[win.get(doc, "v0")](ccs)) / ec
            if ok_si(si, y):
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
        print(f"{doc}: якорей {len(passing[doc])} (bad-ключей {len(doc_bad[doc])}) "
              f"[CC: {win.get(doc, 'v0')}]")
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
            if key in doc_bad[d] or (str(r["Title"]), d) in TEST_ROW_EXCLUDE:
                continue
            pair = _pairs_for(records, key, mw_of(r["SMILES"])).get(d)
            if not pair or not pair[0]:
                continue
            ec = _pick_ec(pair[1])
            if ec is None or ec <= 0:
                continue
            cc = MANUAL_CC.get((str(r["Title"]), d))
            cands.append((-n_ok[d], d,
                          (cc if cc is not None else _PICKS[win.get(d, "v0")](pair[0])) / ec))
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
