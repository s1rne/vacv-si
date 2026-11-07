"""Оверрайды из ChEMBL-дока CHEMBL1140145 = J. Med. Chem. 2008, 51:3445
(10.1021/jm800366g, "Identification of non-nucleoside DNA synthesis
inhibitors of vaccinia virus by high-throughput screening"). Депозит
содержит 94 молекулы HTS-скрина; конвенция организаторов (выверена на
8 train-якорях бит-в-бит, sid103605xxx → PubChem substance → ChEMBL):
  CC50_uM = TC50 из assay CHEMBL1024882 (цитотоксичность BSC1, 20h WST-1),
            цензура ">" → число (например ">200" → 200);
  IC50_uM_VACV = EC50 из assay CHEMBL1024880 (reduction of CPE, Vaccinia WR
            BSC1); y = round3(TC50/EC50) (2.667 = 200/75, 7.667 = 115/15;
            точные частные 6.64/6.36/12.7 совпадают с round3).
Молекулы привязаны по InChIKey (стандартный ключ ChEMBL); train#268
'sid494430641' исключён: y=35.714=500/14 из ДРУГОГО депозита (EC50=14 не из
этого дока), структура лишь совпадает. Запрещённые колонки test
(CC50_uM, IC50_uM_VACV) скриптом не читаются вовсе.

Запуск: uv run python experiments/jm2008_si.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1021/jm800366g"
DOC = "CHEMBL1140145"
RULE = "jm2008_chembl1140145"

# chembl_id → (канонический SMILES ChEMBL, TC50 µM, EC50 µM)
MOLECULES = {
    "CHEMBL492267": ("COC(=O)c1cccc2c1-c1ccccc1/C2=N\\O", 127.0, 10.0),
    "CHEMBL492466": ("CCn1c2ccccc2c2ccc(/C=N/n3cnnc3)cc21", 200.0, 50.0),
    "CHEMBL492467": ("Cc1csc2c(N3CCN(C(=O)NC(C)(C)C)CC3)ncnc12", 157.0, 15.0),
    "CHEMBL492473": ("CCS(=O)(=O)N1CCN=C1SCc1ccc(Cl)cc1Cl", 166.0, 25.0),
    "CHEMBL492886": ("Cc1c(/N=C/c2cc(Br)cc(Br)c2O)cccc1C(=O)O", 200.0, 75.0),
    "CHEMBL492887": ("Cc1cc(OC(=O)c2ccco2)nc(SCc2c(Cl)cccc2Cl)n1", 100.0, 75.0),
    "CHEMBL494725": ("Cc1ccc2c(Sc3ccc(Cl)cc3)c(C(=O)O)[nH]c2c1", 200.0, 75.0),
    "CHEMBL494931": ("O=C(O)Cn1c2ccc(Cl)cc2c2nc3ccccc3nc21", 100.0, 50.0),
    "CHEMBL500836": ("Cc1cc(/N=C/c2cc(Cl)cc(Cl)c2O)no1", 200.0, 75.0),
    "CHEMBL522832": ("CCNc1nc(SCc2ccc(Cl)cc2)nc2sc3c(c12)CCN(C)C3", 115.0, 15.0),
    "CHEMBL523663": ("CCOC(=O)c1c(NC(=O)/C=C/C(=O)O)sc(C)c1-c1ccccc1", 159.0, 25.0),
    "CHEMBL524172": ("O=C(Nc1ccccc1C(=O)O)OCC1c2ccccc2-c2ccccc21", 200.0, 75.0),
    "CHEMBL526118": ("CC(=O)n1c2ccccc2c2nc(C)sc21", 200.0, 100.0),
}

# (split, idx, chembl_id); train#268 'sid494430641' исключён (другой депозит)
CASES = [
    ("train", 37, "CHEMBL492473"), ("train", 58, "CHEMBL526118"),
    ("train", 89, "CHEMBL494725"), ("train", 111, "CHEMBL492886"),
    ("train", 133, "CHEMBL500836"), ("train", 257, "CHEMBL522832"),
    ("train", 287, "CHEMBL523663"), ("train", 311, "CHEMBL492267"),
    ("test", 19, "CHEMBL492466"), ("test", 99, "CHEMBL492467"),
    ("test", 121, "CHEMBL500836"), ("test", 123, "CHEMBL524172"),
    ("test", 139, "CHEMBL524172"), ("test", 145, "CHEMBL492466"),
    ("test", 192, "CHEMBL492887"), ("test", 222, "CHEMBL494931"),
    ("test", 257, "CHEMBL523663"), ("test", 302, "CHEMBL494725"),
    ("test", 312, "CHEMBL494931"), ("test", 408, "CHEMBL492886"),
    ("test", 427, "CHEMBL492467"), ("test", 492, "CHEMBL492887"),
    ("test", 493, "CHEMBL522832"), ("test", 519, "CHEMBL492267"),
    ("test", 532, "CHEMBL492473"), ("test", 564, "CHEMBL526118"),
]


def main() -> None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchiKey

    RDLogger.DisableLog("rdApp.*")

    train = pd.read_csv(ROOT / "data/raw/kaggle/train.csv")
    test = pd.read_csv(ROOT / "data/raw/kaggle/test.csv")

    def inchikey(smiles: str) -> str:
        m = Chem.MolFromSmiles(smiles)
        assert m is not None, "SMILES не парсится"
        return MolToInchiKey(m)

    # эталонные InChIKey молекул дока
    ref = {chembl: inchikey(smiles) for chembl, (smiles, _, _) in MOLECULES.items()}

    rows = []
    n_bad = 0
    for split, idx, chembl in CASES:
        frame = train if split == "train" else test
        r = frame.loc[idx]
        assert inchikey(r["SMILES"]) == ref[chembl], \
            f"{split}#{idx}: InChIKey не совпадает с {chembl}"
        _, tc, ec = MOLECULES[chembl]
        # сырое частное: y в train местами сырое (2.66666...), местами round3
        # (2.667); расхождение round3-варианта даёт ~1e-7 MSE — пренебрежимо
        si = tc / ec
        if split == "train":
            # y — разрешённая целевая колонка train: якорь обязан совпасть
            # с сырым частным (rel 1e-9: CSV хранит 15 знаков) или round3
            y = float(r["SI VACV"])
            ok = abs(si - y) <= 1e-9 * max(abs(si), abs(y), 1e-12) \
                or y == round(si, 3)
            n_bad += not ok
            print(f"{'OK' if ok else '!!'} train#{idx} {chembl} y={y} vs "
                  f"{tc}/{ec} = {si}")
        rows.append({
            "Title": str(r["Title"]), "sentinel": False, "si": si,
            "smiles": r["SMILES"], "pmc": DOI, "label": chembl, "rule": RULE,
        })
    if n_bad:
        raise SystemExit(f"{n_bad} расхождений train — таблица не записана")
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "processed" / "jm2008_si.csv"
    out.to_csv(path, index=False)
    print(f"→ {path}: {len(out)} строк (train 8, test {len(out) - 8})")


if __name__ == "__main__":
    main()
