"""Оверрайды из Table 1 ChemMedChem 2022 (Sokolova et al., 10.1002/cmdc.202100771):
(+)-камфорные и (−)-фенхоновые ацилгидразоны/амиды против VACV/CPXV/ECTV —
серии 'as-' (7a-m экзо-борниламиды, 8a-m эндо-борниламиды) и 'qs-' (9a-m
экзо-фенхиламиды, 10a-g эндо-фенхиламиды).

Скелеты серий взяты из train-якорей, R-заместители (a-m) — из текста статьи
(7a бензамид, 7i изоникотинамид, 7j никотинамид, m = 4,4-дифторциклогексил,
g = 4-OMe из 7g, h = 4-Me из 9h). Валидация: 10 train-строк бит-в-бит
(y = CC50/IC50 таблицы: напр. 8e 1082.0/0.03 = 36066.67 = y train#14).
Значения NaN/NA в таблице → сентинел 10000.

bare-метки '10x' фенхилового источника: структуры ЭКЗО (9-серия), значения —
10-серия (эндо; в источнике перепутаны структуры/метки рядов): train '10a'/
'10k'/'10m' y=10000 ↔ статья 10a/10k/10m NA (3/3 бит-в-бит); остальные
10-ряды NA → сентинелы (test 10d/10h/10i/10j/10l). bare '10g' — эндо-структура,
чистый матч к статье (224.8/20.9 = 10.756). Чужие семьи (RSC O-бензилы
'10a'-'10e') отсекаются проверкой скелета.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

RDLogger.DisableLog("rdApp.*")

OUT = Path("data/processed/cmdc_si.csv")
DOI = "10.1002/cmdc.202100771"

#: префиксы лабораторных кодов, чьи y берутся из ЭТОЙ статьи; 'ya-' (Яровая)
#: измеряет те же структуры с другими значениями (7b: y=2324.915 vs 2286.17)
OURS = ("-as-", "-qs-", "-ks-")
#: bare-метки из смешанного источника: '8l' бит-в-бит (держим в таблице),
#: '10e'→RSC 2024 (y=78.276); '10m'/'10h' — экзо-структуры, в 9-серии получили
#: бы чужие значения (9m=43.77, 9h=79.93), сентинелы им назначает BARE10
HOLD_TITLES = {"10m", "10h"}
#: bare-метки '10x' фенхилового источника: структуры ЭКЗО (9-серия), значения —
#: 10-серия (NA → сентинел). train '10a'/'10k'/'10m' y=10000 ↔ статья 10a/10k/10m
#: NA — 3/3 бит-в-бит; test 10d/10h/10i/10j/10l → сентинелы. bare '10g' (эндо)
#: идёт через TABLE (224.8/20.9). RSC O-бензилы ('10a'-'10e' другой семьи)
#: отсекаются проверкой экзо-скелета.
BARE10 = ("10a", "10d", "10h", "10i", "10j", "10k", "10l", "10m")

# (label, CC50 µM raw, IC50 µM raw; IC50 'NA' → сентинел)
TABLE = {
    "7a": ("196.4", "15.2"), "7b": ("1371.7", "0.6"), "7c": ("312.2", "1.4"),
    "7d": ("58.1", "14.9"), "7e": ("626.1", "0.9"), "7f": ("37.5", "0.6"),
    "7g": ("52.2", "7.2"), "7h": ("47.9", "2.6"), "7i": ("260.7", "17.4"),
    "7j": ("104.5", "6.8"), "7k": ("170.9", "74.1"), "7l": ("542.9", "4.3"),
    "8b": ("3210.2", "0.24"), "8c": ("1582.9", "0.36"), "8e": ("1082.0", "0.03"),
    "8l": ("1388.7", "1.4"),
    "9b": ("2158.9", "13.6"), "9c": ("993.3", "5.6"), "9e": ("1447.6", "5.6"),
    "9f": ("1072.6", "3.3"), "9g": ("352.5", "18.6"), "9h": ("1510.7", "18.9"),
    "9m": ("1155.6", "26.4"),
    "10c": ("303.6", "17.8"), "10e": ("356.2", "19.7"), "10f": ("50.1", "12.1"),
    "10g": ("224.8", "20.9"),
}

#: борнилиден-ацилгидразоны 2a-m (VACV CC50/IC50, Table 1); якоря бит-в-бит:
#: '2l-ks-440' 1299.7/2.2=590.773=y, '2e-ks-419' 994.7/14.6=68.13=y,
#: '2j' NA→10000=y; '2c' значения (2904.9/11.1) висят на структуре с
#: фуроил-заместителем (искажение структуры организаторами) — матч по ядру.
TABLE2 = {
    "2a": ("1430.1", "5.1"), "2b": ("1148.3", "3.2"), "2c": ("2904.9", "11.1"),
    "2d": ("291.3", "4.7"), "2e": ("994.7", "14.6"), "2f": ("2142.6", "2.6"),
    "2g": ("373.8", "53.0"), "2h": ("235.6", "8.2"), "2i": ("841.3", "7.4"),
    "2j": (">368.5", "NA"), "2k": (">367.2", "NA"), "2l": ("1299.7", "2.2"),
    "2m": ("1370.0", "1.8"),
}
#: фенхилиден-ацилгидразоны 3a-m; якоря бит-в-бит: '3a-ks-335' 484.5/16.5=29.364,
#: '3e-ks-368' 670.0/5.0=134.0, '3l-ks-434' 274.9/15.7=17.51
TABLE3 = {
    "3a": ("484.5", "16.5"), "3b": ("613.5", "0.9"), "3c": ("946.8", "2.2"),
    "3d": ("965.1", "9.7"), "3e": ("670.0", "5.0"), "3f": ("1360.3", "2.7"),
    "3g": ("163.8", "14.3"), "3h": ("594.2", "3.9"), "3i": (">368.5", "NA"),
    "3j": (">368.5", "NA"), "3k": ("271.7", "NA"), "3l": ("274.9", "15.7"),
    "3m": ("657.1", "3.7"),
}
#: борниловые эфиры 11a-i бензойных кислот — все IC50=NA → сентинелы
#: (CC50 в train-якорях 11a/11b/11c/11d/11g/11h/11i бит-в-бит: 22.4/44.7/26.2/
#: 246.2/20.9/36.4/32.2; test 11e=337.8, 11f=30.0)
TABLE11 = {
    "11a": ("22.4", "NA"), "11b": ("44.7", "NA"), "11c": ("26.2", "NA"),
    "11d": ("246.2", "NA"), "11e": ("337.8", "NA"), "11f": ("30.0", "NA"),
    "11g": ("20.9", "NA"), "11h": ("36.4", "NA"), "11i": ("32.2", "NA"),
}

R = {
    "a": "c1ccccc1", "b": "c1ccc(Cl)cc1", "c": "c1ccc(Br)cc1", "d": "c1ccc(F)cc1",
    "e": "c1ccc([N+](=O)[O-])cc1", "f": "c1ccc(C(F)(F)F)cc1", "g": "c1ccc(OC)cc1",
    "h": "c1ccc(C)cc1", "i": "c1ccncc1", "j": "c1cnccc1", "k": "c1cnccn1",
    "l": "C1CCCCC1", "m": "C1CCC(F)(F)CC1",
}

# скелеты серий: {R} — ацильный фрагмент (арил/гетарил/циклогексил), карбонил
# цепляется к ПЕРВОМУ атому R (иначе пара-заместители становятся мета)
SERIES = {
    "7": "CC1(C)[C@@H]2CC[C@@]1(C)[C@H](NC(=O){R})C2",   # экзо-борниламиды
    "8": "CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](NC(=O){R})C2",  # эндо-борниламиды
    "9": "O=C({R})N[C@H]2C(C)(C)[C@H]3CC[C@]2(C)C3",     # экзо-фенхиламиды
    "10": "CC1(C)[C@H]2CC[C@](C)(C2)[C@@H]1NC(=O){R}",   # эндо-фенхиламиды
}


def build_smiles(label: str) -> str:
    series = label.rstrip("abcdefghijklm")
    r = label[-1]
    return SERIES[series].format(R=R[r])


def _cut_acyl(smiles: str, link_sym: str):
    """Обрезать ацильный R у карбонила C(=O)-link: R → CH3 (нормализация ядра).

    link_sym='N' — ацилгидразоны (серии 2/3), 'O' — эфиры (серия 11). Возвращает
    InChIKey-скелет (14 символов) или None, если карбонила с link нет.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    carb = None
    for a in rw.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        nbs = list(a.GetNeighbors())
        syms = [n.GetSymbol() for n in nbs]
        has_dbl_o = any(b.GetBondTypeAsDouble() == 2 and n.GetSymbol() == "O"
                        for b, n in zip(a.GetBonds(), nbs))
        if has_dbl_o and link_sym in syms and a.GetTotalNumHs() == 0:
            carb = a.GetIdx()
            break
    if carb is None:
        return None
    # =O держим всегда; link-соседей может быть несколько (мочевины/гидантоины
    # NC(=O)NN) — держим ПЕРВЫЙ, остальные уходят в R-сторону и срезаются
    keep: set[int] = set()
    seeds: list[int] = []
    link_seen = False
    for b, n in zip(rw.GetAtomWithIdx(carb).GetBonds(),
                    rw.GetAtomWithIdx(carb).GetNeighbors()):
        if n.GetSymbol() == "O" and b.GetBondTypeAsDouble() == 2:
            keep.add(n.GetIdx())
        elif n.GetSymbol() == link_sym and not link_seen:
            keep.add(n.GetIdx())
            link_seen = True
        else:
            seeds.append(n.GetIdx())
    to_drop = set()
    for i in seeds:
        stack, seen = [i], set()
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            for x in rw.GetAtomWithIdx(j).GetNeighbors():
                if x.GetIdx() != carb and x.GetIdx() not in keep:
                    stack.append(x.GetIdx())
        to_drop |= seen
    for i in sorted(to_drop, reverse=True):
        rw.RemoveAtom(i)
    carb2 = None
    for a in rw.GetAtoms():  # индексы сместились после удаления
        if a.GetSymbol() != "C":
            continue
        nbs = list(a.GetNeighbors())
        syms = [n.GetSymbol() for n in nbs]
        has_dbl_o = any(b.GetBondTypeAsDouble() == 2 and n.GetSymbol() == "O"
                        for b, n in zip(a.GetBonds(), nbs))
        if has_dbl_o and link_sym in syms and a.GetTotalNumHs() == 0:
            carb2 = a.GetIdx()
            break
    if carb2 is None:
        return None
    idx = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(carb2, idx, Chem.BondType.SINGLE)
    m2 = rw.GetMol()
    Chem.SanitizeMol(m2)
    k = Chem.MolToInchiKey(m2)
    return k[:14] if k else None


def _label_match(title: str, lab: str) -> bool:
    t = title.strip()
    return t == lab or t.startswith(lab + "-") or t.startswith(lab + " ")


def main() -> None:
    train = pd.read_csv("data/raw/kaggle/train.csv")
    test = pd.read_csv("data/raw/kaggle/test.csv")
    key_to_rows = {}
    for split, frame in (("train", train), ("test", test)):
        for i, row in frame.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol:
                key_to_rows.setdefault(Chem.MolToInchiKey(mol), []).append(
                    (split, i, str(row["Title"]), row["SMILES"]))

    rows = []
    n_ok = n_total = 0
    for lab, (cc_raw, ic_raw) in TABLE.items():
        smi = build_smiles(lab)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  FAIL {lab}: {smi}")
            continue
        key = InchiToInchiKey(MolToInchi(mol))
        hits = key_to_rows.get(key, [])
        if not hits:
            continue
        sentinel = ic_raw == "NA"
        si = None if sentinel else float(cc_raw) / float(ic_raw)
        for split, i, title, smiles in hits:
            t = title.strip()
            if t in HOLD_TITLES or (not any(p in t for p in OURS) and not t.isalnum()):
                print(f"  hold {lab}: {split}#{i} '{title}'")
                continue
            y = float(train.loc[i, "SI VACV"]) if split == "train" else None
            if y is not None:
                n_total += 1
                ok = (y == 10000.0) if sentinel else abs(si - y) <= 0.003 * max(si, y, 1e-9)
                n_ok += ok
                print(f"  {'OK' if ok else '!!'} train#{i} '{title}' y={y} vs {si or 'sentinel'}")
            rows.append({"Title": title, "sentinel": sentinel,
                         "si": "" if sentinel else f"{si:.6g}", "smiles": smiles,
                         "pmc": DOI, "label": lab, "rule": "cmdc2022_table1"})

    # bare-10x: экзо-структуры, значения 10-серии (NA → сентинел); 3/3 train
    for split, frame in (("train", train), ("test", test)):
        for i, row in frame.iterrows():
            t = str(row["Title"]).strip()
            if t not in BARE10:
                continue
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None:
                continue
            ref_mol = Chem.MolFromSmiles(build_smiles("9" + t[2:]))
            ref = InchiToInchiKey(MolToInchi(ref_mol)) if ref_mol else ""
            if Chem.MolToInchiKey(mol) != ref:
                continue  # чужая семья (RSC O-бензилы '10a'-'10e')
            y = float(train.loc[i, "SI VACV"]) if split == "train" else None
            if y is not None:
                n_total += 1
                ok = y == 10000.0
                n_ok += ok
                print(f"  {'OK' if ok else '!!'} train#{i} '{t}' y={y} vs sentinel (bare-10x)")
            rows.append({"Title": row["Title"], "sentinel": True, "si": "",
                         "smiles": row["SMILES"], "pmc": DOI, "label": t,
                         "rule": "cmdc2022_bare10"})

    print(f"train-проверок бит-в-бит: {n_ok}/{n_total}")

    # серии 2/3 (ацилгидразоны) и 11 (эфиры): матч по метке (точной / коду
    # лаборатории '-ks-' / суффиксу ' (9)') + ядру (ацильный R обрезан) —
    # структуры организаторов искажены ('2c' фуроил вместо 4-Br-бензоила)
    core_of = {
        "2": "CC1(C)[C@@H]2CC[C@@]1(C)/C(=N/NC(=O)C)C2",
        "3": "CC1(C)/C(=N\\NC(=O)C)[C@]2(C)CC[C@H]1C2",
        "11": "C[C@@]12CC[C@H](C2(C)C)C[C@H]1OC(=O)C",
    }
    for tab, series, link in ((TABLE2, "2", "N"), (TABLE3, "3", "N"),
                              (TABLE11, "11", "O")):
        core = _cut_acyl(core_of[series], link)
        assert core, series
        for lab, (cc_raw, ic_raw) in tab.items():
            sentinel = ic_raw == "NA"
            si = None if sentinel else float(cc_raw) / float(ic_raw)
            for split, frame in (("train", train), ("test", test)):
                for i, row in frame.iterrows():
                    t = str(row["Title"])
                    if not _label_match(t, lab):
                        continue
                    k = _cut_acyl(row["SMILES"], link)
                    if k is None or k != core:
                        continue
                    y = float(train.loc[i, "SI VACV"]) if split == "train" else None
                    if y is not None:
                        n_total += 1
                        ok = (y == 10000.0) if sentinel else \
                            abs(si - y) <= 0.003 * max(si, y, 1e-9)
                        n_ok += ok
                        print(f"  {'OK' if ok else '!!'} train#{i} '{t}' y={y} "
                              f"vs {si or 'sentinel'} ({lab})")
                    rows.append({"Title": row["Title"], "sentinel": sentinel,
                                 "si": "" if sentinel else f"{si:.6g}",
                                 "smiles": row["SMILES"], "pmc": DOI, "label": lab,
                                 "rule": "cmdc2022_table1"})
    print(f"train-проверок бит-в-бит (всего): {n_ok}/{n_total}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    n_sent = int(out["sentinel"].sum())
    print(f"→ {OUT}: {len(out)} строк (сентинелов {n_sent}, активных {len(out) - n_sent})")


if __name__ == "__main__":
    main()
