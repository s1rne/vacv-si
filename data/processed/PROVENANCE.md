# Происхождение таблиц в data/processed

Файл `competition_open_data_lookup.csv` собран командой:

```
uv run python -m vacv.cli coverage
```

Код: коммит `9e3c050` (правило восстановления подбирается только по целевой колонке `SI VACV`, колонки CC50_uM и IC50_uM_VACV не читаются).

Промежуточные таблицы `data/interim` в репозиторий не входят, поэтому их контрольные суммы записаны здесь:

| файл | sha256 |
|---|---|
| `chembl_targeted.csv.gz` | `d6570bfa1eb9ec12` |
| `chembl.csv.gz` | `da23cc1f3e464b53` |
| `competition_molecules.csv` | `6a5694e5ede5c5cf` |
| `curated.csv.gz` | `5cabd5b6149c8058` |
| `inactive.csv.gz` | `b53f640f8dc78b1f` |
| `measurements.csv.gz` | `7c2da980d484b6ad` |
| `pubchem.csv.gz` | `1c9474c15f5a7ae5` |
