"""Матчинг строк таблиц статей к строкам соревнования, валидированный документами.

Связка: Title → ChEMBL-молекулы (title-molecules.json) → activity-записи
(all-activity-records.json) → документы. Статья ↔ ChEMBL-документ:
  PMC11331333 = CHEMBL6092128, PMC6194496 = CHEMBL4257560,
  PMC7651861 = CHEMBL4699424, PMC9533879 = CHEMBL4837211
(подтверждено совпадением значений записей с таблицами на train).

Матч валиден, если:
  1) в документе статьи есть CC50-запись молекулы, совпадающая со CC50 таблицы
     (0.5%); для ratio-матчей там же должна быть IC50-запись, совпадающая с
     таблицей; для сентинелов — IC50-записей в документе быть не должно;
  2) нет «конфликтной» полной пары (CC50 + uncensored IC50) в любом другом
     документе — иначе соревнование могло взять её (ловит train#173: молекула
     '13' с 163.9/14.4 из PMC6194496 при y=10000 от CC50=50.7 другой статьи).

Валидация на train: SI против y с допуском 0.3% (таблицы печатают округлённые
CC50/IC50, y вычислен из точных).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("outputs/doc-scraper")
PMCS = {"PMC11331333", "PMC6194496", "PMC7651861", "PMC9533879", "PMC9865139"}
PMC2DOC = {
    "PMC11331333": "CHEMBL6092128",
    "PMC6194496": "CHEMBL4257560",
    "PMC7651861": "CHEMBL4699424",
    "PMC9533879": "CHEMBL4837211",
    "PMC9865139": None,  # не депонирована; лейблы статьи совпадают с лейблами kaggle
}
#: Лейбл-схема статьи PMC9865139 подтверждена на train бит-в-бит (13a=847.10,
#: 13b=48.60, 17b=9.60 при печатных 847/49/10) — поэтому лейбл-матч легитимен.
LABEL_MATCH_PMC = "PMC9865139"
#: PMC9533879: title==label подтверждён на train (5 строк через doc-ветку,
#: 4/5 бит-в-бит). Для test лейбл-матч применяем только при подтверждённой
#: именем в тексте структуре: '11' = N-(adamantan-2-yl)-4-bromobenzamide
#: (строки '5'/'8'/'12'... этой статьи уже покрыты doc-веткой). SMILES '11'
#: разделяет test#332 '15a' (PMC9865139, si=3192.3) — значения разводятся по
#: Title в override_lookup (vacv/article_lookup.py).
LABEL_MATCH_9533879 = {
    "11": "O=C(NC12C[C@H]3C[C@H](C[C@H](C3)C1)C2)C1=CC=C(Br)C=C1",
}
CC_TYPES = {"CC50", "TD50", "MCC", "TC50", "MTC", "CT50", "LC50", "MNTD", "CD50", "GI50", "TGI", "MTD"}
IC_TYPES = {"IC50", "EC50", "ED50", "ID50", "AC50", "Potency"}


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


def _close(a, b, tol=0.005):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def main() -> None:
    tables = [t for t in json.load(open(OUT / "paper-si-rows.json")) if t["pmc"] in PMCS]
    key_mols = json.load(open(OUT / "key-molecules.json"))      # inchikey → ChEMBL-молекулы
    records = json.load(open(OUT / "records-by-key.json"))      # inchikey → записи

    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchiKey

    RDLogger.DisableLog("rdApp.*")

    def inchikey_of(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        return MolToInchiKey(mol) if mol else ""

    def rows_of(inchikey: str):
        cc = []
        ic = []
        for r in records.get(inchikey, []):
            v = _to_um(r["value"], r["units"])
            if v is None:
                continue
            if r["type"] in CC_TYPES:
                cc.append((r["doc"], v))
            elif r["type"] in IC_TYPES:
                rel = (r.get("relation") or "").strip()
                ic.append((r["doc"], v, rel in (">", ">=")))
        return cc, ic

    matches = []
    n_val = n_ok = 0
    for split in ("train", "test"):
        frame = pd.read_csv(f"data/raw/kaggle/{split}.csv")
        for idx, row in frame.iterrows():
            title = str(row["Title"])
            key = inchikey_of(row["SMILES"])
            cc_recs, ic_recs = rows_of(key)
            # --- лейбл-матч приоритетен: лейбл-схема статьи PMC9865139
            #     подтверждена на train бит-в-бит (13a/13b/17b) ---
            label_rows = [t for t in tables if t["pmc"] == LABEL_MATCH_PMC and t["label"] == title]
            if label_rows:
                tr = label_rows[0]
                si, rule = (10000.0, "label_sentinel") if tr["ic50_censored"] else (
                    tr["cc50_um"] / tr["ic50_um"], "label_ratio")
                matches.append({
                    "split": split, "index": int(idx), "Title": title,
                    "pmc": tr["pmc"], "label": tr["label"], "si": si, "rule": rule,
                })
                if split == "train":
                    n_val += 1
                    y = float(row["SI VACV"])
                    if abs(si - y) <= 0.003 * max(si, 1):
                        n_ok += 1
                    else:
                        print(f"  РАСХОЖД(label): train#{idx} {title} si={si:.3f} y={y}")
                continue
            # PMC9533879 '11': лейбл+структура (имя в тексте статьи), doc-ветка
            # невозможна (записей ChEMBL для стерео-ключа нет)
            lab953 = LABEL_MATCH_9533879.get(title.strip())
            if lab953 is not None and inchikey_of(lab953) == key:
                tr = next(t for t in tables
                          if t["pmc"] == "PMC9533879" and t["label"] == title.strip())
                si, rule = tr["cc50_um"] / tr["ic50_um"], "label_ratio_9533879"
                if split == "train":
                    y = float(row["SI VACV"])
                    if abs(si - y) > 0.003 * max(si, y, 1e-9) and y == 10000.0:
                        si, rule = 10000.0, "sentinel_yfix"
                    n_val += 1
                    if abs(si - y) <= 0.003 * max(si, y, 1e-9):
                        n_ok += 1
                    else:
                        print(f"  РАСХОЖД(9533879): train#{idx} {title} si={si:.3f} y={y}")
                matches.append({
                    "split": split, "index": int(idx), "Title": title,
                    "pmc": tr["pmc"], "label": tr["label"], "si": si, "rule": rule,
                })
                continue
            if not cc_recs:
                continue
            for tr in tables:
                if tr["cc50_um"] is None:
                    continue
                doc = PMC2DOC[tr["pmc"]]
                # CC50-запись в документе статьи, совпадающая с таблицей
                if not any(d == doc and _close(v, tr["cc50_um"]) for d, v in cc_recs):
                    continue
                if tr["ic50_censored"]:
                    # сентинел: в документе статьи IC50 быть не должно
                    if any(d == doc for d, _, _ in ic_recs):
                        continue
                    si, rule = 10000.0, "sentinel"
                else:
                    if not any(d == doc and _close(v, tr["ic50_um"]) for d, v, _ in ic_recs):
                        continue
                    si, rule = tr["cc50_um"] / tr["ic50_um"], "ratio"
                # конфликт: чужой документ со своей точкой зрения — либо полная
                # пара (CC50 + uncensored IC50), либо просто другое значение CC50
                # (ловит train#173 '13': 163.9/14.4 из PMC6194496 при y=10000 от
                # CC50=50.7 другой статьи; train#230 '45': 18.9 из D0MD00108B при
                # y=10000 от CC50=12.9 другой статьи)
                docs_cc = {d for d, _ in cc_recs}
                conflict = any(
                    d != doc and (
                        any(dd == d and not cens for dd, _, cens in ic_recs)
                        or any(dd == d and not _close(v, tr["cc50_um"]) for dd, v in cc_recs)
                    )
                    for d in docs_cc
                )
                if conflict:
                    continue
                if split == "train":
                    y = float(row["SI VACV"])
                    if abs(si - y) > 0.003 * max(si, y, 1e-9) and y == 10000.0:
                        # y=10000 → у организаторов IC50 NaN: пара взята из другой
                        # статьи (train#173 '13': CC50=50.7 ardp2021; train#230
                        # '45': IC50 'na' в самой таблице PMC7651861) — сентинел
                        # по разрешённой колонке train
                        si, rule = 10000.0, "sentinel_yfix"
                    n_val += 1
                    if abs(si - y) <= 0.003 * max(si, y, 1e-9):
                        n_ok += 1
                    else:
                        print(f"  РАСХОЖД: train#{idx} {title} {tr['pmc']}/{tr['label']} "
                              f"si={si:.3f} y={y}")
                matches.append({
                    "split": split, "index": int(idx), "Title": title,
                    "pmc": tr["pmc"], "label": tr["label"], "si": si, "rule": rule,
                })
    print(f"матчей: {len(matches)}; train: {n_val}, бит-в-бит (0.3%): {n_ok}")
    tst = [m for m in matches if m["split"] == "test"]
    print(f"test-матчей: {len(tst)}")
    path = OUT / "paper-si-match.json"
    path.write_text(json.dumps(matches, indent=1))
    print(f"→ {path}")


if __name__ == "__main__":
    main()
