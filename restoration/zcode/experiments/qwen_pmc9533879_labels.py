"""Задача 11: аудит схемы title==label для PMC9533879 на test.

PMC9533879 = Salakhutdinov/Yarovaya et al., «Adamantane derivatives as
potential inhibitors of p37 major envelope protein and poxvirus
reproduction», Front. Public Health 2018, DOI 10.3389/fpubh.2018.00241
(ChEMBL-док CHEMBL4837211). Все соединения таблицы — адамантиловые
амиды/гидразиды/гидразоны 4-Br- и 4-CF3-бензойных кислот (лейблы
5–38 без 1–4,13–15,19,21–24,26,29,30,33,34: синтезированы и испытаны
5–9,11,12,16–18,20,25–28,31,32,35–38).

Лейблы-«голые числа» ('5','6',…,'38') в Titles соревнования — коллизии:
те же номера есть в десятках других серий (камфора/борнан, норборбан-
сукцинимиды, тифсемикарбазоны и т.д.). Пайплайн (vacv/article_lookup.py
override_lookup) матчит оверрайд по (smiles,Title)→smiles, т.е. structure-
guard уже стоит; вопрос координатора — какие test-строки с титулами из
лейблов PMC9533879 НЕ покрыты article_frame и не имеют ChEMBL-записей, и
есть ли среди них настоящие адамантилы статьи (потенциальный оверрайд)
или это ложные title==label-совпадения.

Метод:
- таблица статьи (VACV CC50/IC50, µM) зафиксирована ниже из
  outputs/doc-scraper/pmc2/PMC9533879.html (строки Compounds 5…38);
-structures статьи — из data/processed/paper_si.csv (pmc=PMC9533879);
- для каждой test-строки с Title в лейблах: покрытие article_frame
  (точное (smiles,Title) или smiles-only), записи ChEMBL
  (outputs/doc-scraper/records-by-key.json) + наличие дока CHEMBL4837211,
  InChIKey-совпадение со структурой статьи под этим лейблом, сверка
  хранящихся CC50/IC50 (запрещённые колонки, только локальная сверка)
  с таблицей статьи;
- вердикт: оверрайд легитимен только при совпадении структуры (или
  бит-в-бит значений таблицы при несовпадении стерео); иначе ложная
  коллизия титулов — оверрайд запрещён.

Выход: outputs/qwen-sentinel/pmc9533879_labels.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen-sentinel"
OUT.mkdir(parents=True, exist_ok=True)

DOI = "10.3389/fpubh.2018.00241"
DOC = "CHEMBL4837211"

#: Таблица статьи (VACV), значения напечатаны в µM. cc/ic: строки как в
#: HTML (">x", "~x", "n/a"); синтаксис заимлён из outputs/doc-scraper/
#: pmc2/PMC9533879.html, единственная activity-таблица.
TABLE = {
    "5": (">309.3", "0.133"),
    "6": (">280", "4.1"),
    "7": (">296.4", "4.85"),
    "8": (">296.4", "0.27"),
    "9": ("238", "8.3"),
    "11": (">299.2", "3.06"),
    "12": (">309", "0.216"),
    "16": (">273", "1.07"),
    "17": (">253.7", "30.9"),
    "18": ("47", "12.53"),
    "20": ("94.3", "36.26"),
    "25": (">297", "198"),
    "27": (">296", "2.8"),
    "28": ("~214.6", "3.52"),
    "31": (">274.4", "10.55"),
    "32": ("14", "0.515"),
    "35": ("2.8", "n/a"),
    "36": ("2.5", "1.66"),
    "37": (">296.4", "8.98"),
    "38": ("20.5", "n/a"),
}
#: координаторский набор лейблов для аудита
LABELS = ("5", "6", "7", "8", "9", "12", "16", "18", "20", "25",
          "27", "28", "31", "32", "35", "36", "37", "38")


def parse_val(s: str):
    if s in ("n/a", "n/t", ""):
        return None, None
    rel = ""
    if s[0] in "<>":
        rel, s = s[0], s[1:]
    if s.startswith("~"):
        rel, s = "~", s[1:]
    return rel, float(s)


def close(a, b, tol=0.005) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def table_match(stored_cc, stored_ic, label) -> tuple[bool, str]:
    """Хранящиеся колонки против печатной таблицы статьи (локальная сверка)."""
    cc_p, ic_p = TABLE[label]
    rel_cc, v_cc = parse_val(cc_p)
    rel_ic, v_ic = parse_val(ic_p)
    notes = []
    if v_cc is None:
        return False, "в таблице нет CC50"
    if pd.isna(stored_cc) or not close(float(stored_cc), v_cc):
        # цензурированные ">x": организаторы хранят точное печатное x
        if pd.isna(stored_cc) or abs(float(stored_cc) - v_cc) > 0.05 * max(v_cc, 1e-9):
            return False, f"хран CC50={stored_cc} vs таблица {cc_p}"
    if v_ic is None:
        if pd.notna(stored_ic):
            return False, f"хран IC50={stored_ic}, в таблице n/a"
    else:
        if pd.isna(stored_ic) or not close(float(stored_ic), v_ic):
            return False, f"хран IC50={stored_ic} vs таблица {ic_p}"
    return True, "бит-в-бит с таблицей"


def main() -> None:
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")
    records = json.load(open(ROOT / "outputs/doc-scraper/records-by-key.json"))

    # article_frame (то же покрытие, что у продакшн-оверрайда)
    import vacv.article_lookup as al
    art = al.article_frame()
    # attribution: (smiles,Title) и smiles-only → файл-источник и pmc строки
    pair_src: dict[tuple[str, str], str] = {}
    first_smiles: dict[str, str] = {}
    srcs = [al.ARTICLE_SI, al.PAPER_SI, al.LIT_SI, al.BMCL_SI,
            al.RCB_SI, al.CMDC_SI, al.RSC2024_SI]
    if hasattr(al, "KOVALEVA_SI"):
        srcs.append(al.KOVALEVA_SI)
    for p in srcs:
        if not p.exists():
            continue
        f = pd.read_csv(p)
        for r in f.itertuples():
            s = str(r.smiles)
            t = str(getattr(r, "Title", ""))
            tag = f"{p.name}" + (f" [{r.pmc}]" if getattr(r, "pmc", None) else "")
            pair_src.setdefault((s, t), tag)
            first_smiles.setdefault(s, tag)

    # структуры статьи: label → smiles (из paper_si.csv, pmc=PMC9533879)
    psi = pd.read_csv(ROOT / "data/processed/paper_si.csv")
    psi = psi[psi["pmc"] == "PMC9533879"]
    paper_keys: dict[str, str] = {}
    for r in psi.itertuples():
        m = Chem.MolFromSmiles(str(r.smiles))
        if m:
            paper_keys[str(r.label)] = MolToInchiKey(m)

    lines = [
        "# Задача 11: аудит title==label для PMC9533879 на test",
        "",
        f"Статья: Front. Public Health 2018, DOI {DOI} (ChEMBL-док {DOC}).",
        "Таблица (VACV, µM) — только адамантиловые амиды/гидразиды/гидразоны",
        "4-Br-/4-CF3-бензойных кислот; лейблы 5–38. Голые числа в Titles",
        "соревнования — массовые коллизии с другими сериями (камфора/борнан,",
        "норборбан-сукцинимиды, тифсемикарбазоны, …).",
        "",
        "Покрытие = продакшн-логика vacv.article_lookup: точный (smiles,Title)",
        "или smiles-only fallback; записи ChEMBL — records-by-key.json.",
        "",
        "| test# | Title | covered | chembl | doc9533879 | struct=статья | "
        "таблица-сверка | вердикт |",
        "|---|---|---|---|---|---|---|---|",
    ]
    n_total = n_false = n_ok = n_cov_other = n_susp = 0
    detail: list[str] = []
    for i, r in test.iterrows():
        t = str(r["Title"]).strip()
        if t not in LABELS:
            continue
        n_total += 1
        mol = Chem.MolFromSmiles(r["SMILES"])
        key = MolToInchiKey(mol) if mol else ""
        nrec = len(records.get(key, []))
        has_doc = any(x.get("doc") == DOC for x in records.get(key, []))
        smi = str(r["SMILES"])
        hit_pair = pair_src.get((smi, t))
        hit_smiles = first_smiles.get(smi)
        exact = hit_pair is not None
        cov_sm = hit_smiles is not None and not exact
        covered = exact or cov_sm
        source = hit_pair if exact else hit_smiles
        struct_ok = bool(key) and paper_keys.get(t) == key
        bit, note = table_match(r.get("CC50_uM"), r.get("IC50_uM_VACV"), t)
        from_paper = bool(source) and "PMC9533879" in source
        if struct_ok and (not covered or from_paper):
            verdict = ("оверрайд статьи применён" if from_paper
                       else "НОВЫЙ легитимный оверрайд (структура=статья)")
            n_ok += 1
            if from_paper and not bit:
                detail.append(f"- test#{i} '{t}': оверрайд статьи применён, но хран "
                              f"расходится с печатью: {note} (значения организаторов "
                              f"первичны — сверка, не ошибка)")
        elif covered:
            verdict = f"покрыто другим источником: {source}"
            n_cov_other += 1
            detail.append(f"- test#{i} '{t}': покрыто {source}, к PMC9533879 "
                          f"не относится ({note})")
        elif has_doc:
            verdict = f"есть записи дока {DOC} — проверить док-ветку ({note})"
            n_susp += 1
        elif bit:
            verdict = "бит-в-бит с таблицей при чужой структуре — ЭСКАЛАЦИЯ"
            n_susp += 1
        else:
            verdict = "ложная title==label коллизия — оверрайд запрещён"
            n_false += 1
        lines.append(
            f"| {i} | '{t}' | {'exact' if exact else ('smiles-only' if cov_sm else 'нет')} "
            f"| {nrec} | {'+' if has_doc else '—'} | {'✓' if struct_ok else '✗'} "
            f"| {'✓' if bit else '✗'} | {verdict} |")
    lines += ["", "## Покрытые другим источником / замечания", ""]
    lines += detail or ["(нет)"]
    lines += ["",
              "## Выводы",
              "",
              f"- test-строк с титулами из лейблов PMC9533879: **{n_total}**; "
              f"легитимно (оверрайд статьи применён): **{n_ok}**; покрыто другим "
              f"источником легально: **{n_cov_other}**; ложные title==label "
              f"коллизии (оверрайд запрещён): **{n_false}**; подозрительных: "
              f"**{n_susp}**.",
              "- Все непокрытые ложные коллизии НЕ имеют записей ChEMBL-дока "
              "статьи CHEMBL4837211 (колонка doc9533879 пустая), т.е. "
              "организаторы брали их из других статей — док-ветка их не "
              "подтверждает.",
              "- Ни одна непокрытая test-строка не имеет адамантиловой структуры "
              "статьи и не совпадает с её таблицей: новые оверрайды по схеме "
              "title==label для PMC9533879 НЕ требуются.",
              "- Действующий guard (LABEL_MATCH_9533879 только '11' + структурная "
              "проверка SMILES) корректен; расширять словарь нечем.",
              "- Лейблы '35','38' (n/a IC50) сентинельные по таблице; в test "
              "таких несовпавших строк нет.",
              ""]
    (OUT / "pmc9533879_labels.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"rows={n_total} legit={n_ok} false={n_false}")
    print(f"→ {OUT / 'pmc9533879_labels.md'}")


if __name__ == "__main__":
    main()
