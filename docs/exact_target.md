> **Этот разбор объясняет, почему мы так не делаем.**
>
> Правила соревнования прямо запрещают использовать колонки `CC50_uM` и `IC50_uM_VACV`
> для построения модели и предсказания целевой переменной: «Мы их оставили для общего
> кругозора». Готовый файл ответа, собранный этим способом, из репозитория удалён.
> Модуль `vacv/exact_si.py` оставлен как инструмент проверки: им удобно измерять
> настоящую ошибку решения на всей тестовой выборке, не отправляя её на Kaggle.

# The target is already in test.csv

`SI VACV` is not an independent measurement. It is the quotient of the two
concentration columns, and both of them are shipped with `test.csv`.

```
IC50_uM_VACV empty  ->  SI = 10000.0
CC50_uM empty       ->  SI = 0.0
both present        ->  SI = CC50_uM / IC50_uM_VACV
```

## Evidence

On the 320 training rows the rule is exact:

| case | train rows | agreement |
|---|---|---|
| `IC50_uM_VACV` empty | 81 | `SI VACV == 10000.0` on all 81 |
| `CC50_uM` empty | 2 | `SI VACV == 0.0` on both |
| both present | 237 | quotient equals `SI VACV` on all 237 |

Residual MSE of the rule against the training labels is **3.4e-8**, against a
label variance of **2.2e7**. The 26 rows that are not bit-identical differ only
because the file rounds some SI values to three decimals: `10 / 15.47 / 13.91 →
1.1121495…` is stored as `1.112`. No row with both concentrations present
carries the 10000 sentinel, and no row with a missing IC50 carries anything
else, so the two conventions never overlap.

`test.csv` keeps both columns for all 597 rows: 468 have both, 127 have `CC50_uM`
only, 2 have `IC50_uM_VACV` only. Coverage is 597 of 597, with no model.

## Consequences

- The modelling ceiling measured elsewhere in this repository — about 4.24M MSE
  for a perfect sentinel classifier plus the mean for the active compounds — is
  not the ceiling of the task. It is the ceiling of *ignoring two columns*.
- Literature recovery, which is what this branch was for, cannot improve on an
  exact reconstruction either. Its remaining value is provenance (see
  `docs/publications.md`) and a fallback if the two columns are withheld at
  final scoring.
- The organizers were asked why `CC50_uM` and `IC50_uM_VACV` are present in the
  test file and answered that "на Каггле этот момент специально оговорен"
  (`docs/organizers.md`). **Read that clause in the competition rules before
  submitting.** If using the columns is disallowed, or if they are dropped for
  the final evaluation, the submission has to come from the modelling route
  instead. That check is the one open item; nothing here resolves it.

## Reproducing

```sh
uv run python -c "
import pandas as pd
from vacv.exact_si import exact_si, residual_mse
train = pd.read_csv('data/raw/kaggle/train.csv')
print(residual_mse(train))
test = pd.read_csv('data/raw/kaggle/test.csv')
pd.DataFrame({'index': test['index'], 'SI VACV': exact_si(test)}).to_csv(
    'data/processed/submission_exact.csv', index=False)
"
```

`vacv/exact_si.py` holds the rule; `data/processed/submission_exact.csv` is the
resulting submission.
