# Проверка и улучшение базовой модели, 2026-11-07

Предлагаемый профиль — `hybrid`: равная смесь контекстного ExtraTrees и kernel ridge.
Новый файл не отправлялся на Kaggle. Историческое значение 7.49M в README относится
к прежним отправкам и не является измерением качества этого кандидата.

## Данные и границы проверки

- Использованы 320 размеченных строк train и 597 строк test из предоставленного
  архива соревнования. Метки test не использовались. Идентификаторы и порядок
  сабмитов проверены по `sample_submit.csv`.
- `vacv.dataset.load` удаляет `CC50_uM` и `IC50_uM_VACV`; эти столбцы и статистика
  их пропусков не используются моделями. Целевая переменная — только `SI VACV` train.
- Использован существующий снимок разрешённых внешних признаков из ChEMBL/PubChem:
  `competition_open_data_lookup.csv`, `competition_presence.csv`. Повторный сбор
  внешних данных не выполнялся. SHA256 входных CSV сохранены в `data-manifest.json`.
  Данный этап не является повторным независимым аудитом происхождения каждой внешней записи.
- Внешние признаки и литературные значения присоединяются одинаково для CV и
  сабмита. Значение из литературы заменяет модельный прогноз и ограничивается
  диапазоном [0, 10000]. Это существующее правило применяется и на validation.
- Метка неактивности — точное `SI == 10000`. Настоящее значение 36066.667 сохранено
  как активное наблюдение и входит в MSE без обрезки или удаления.

## Протокол

Основная метрика — MSE итогового прогноза на исходной шкале SI. Дополнительно
сохранены MSE до литературной подстановки, ROC AUC и Brier score вероятности
неактивности. AUC не вычисляется по прогнозу после подстановки.

Все случайные проверки — 5-fold `KFold(shuffle=True, random_state=split_seed)`.
Первичный поиск использовал split seed 0, 1, 2 и один model seed (0).
По его результатам выбраны `context_half`, `kernel` и их равная смесь `hybrid`.
Проверка с model seeds 0–2 использовала split seeds 3, 4, 5; финальная проверка
повторила те же splits с model seeds 0–6. Это подтверждение устойчивости на новых
разбиениях тех же строк, а не независимый holdout или nested model selection.
Финальный переход с трёх model seeds к семи не создаёт новые контрольные данные.

Scaffold — пять детерминированных фолдов по Murcko scaffold: группы сортируются
по убыванию размера и по очереди помещаются в самый маленький фолд; пустые остовы
считаются отдельными группами. Scaffolds не пересекаются между train/validation,
но публикационный контекст может пересекаться. Повторение этой же scaffold-схемы
на этапах поиска/подтверждения не является новой независимой проверкой.

Категории, групповые доли, среднее SI активных и коэффициенты модели обучаются
только на тренировочной части соответствующего фолда. Исторический `baseline`
включает собственную обучающую метку строки в групповую долю; validation-метки
туда не входят. Его внешняя CV оценивает именно этот алгоритм, но у него есть
расхождение распределений train/inference. В `crossfit` собственная метка и prior
исключаются через внутренние KFold; результат оказался хуже. `hybrid` не использует
групповое target encoding.

## Точные модели

Все профили используют `p * 10000 + (1 - p) * mean(active_train_SI)`; затем применяется
литературная подстановка. Усреднение seed происходит на уровне вероятностей и
модельных прогнозов. `hybrid` усредняет два компонента с весами 0.5/0.5, без дообучения
весов на confirmation splits. Kernel детерминирован и повторяется без изменений
при каждом seed; дополнительное усреднение влияет только на ExtraTrees.

| Компонент / профиль | Параметры и признаки |
|---|---|
| `baseline` | 0.3 context + 0.7 grouped; структура отдельно имеет вес 0; сглаживание групповой доли 5 |
| ExtraTrees по умолчанию | 800 деревьев, min_samples_leaf=2, max_features=sqrt, n_jobs=-1, random_state=model seed |
| Context | One-hot strain/Culture, пять числовых характеристик Title, matched и девять внешних признаков присутствия |
| Grouped LightGBM | 500 деревьев, learning_rate=0.03, num_leaves=7, min_child_samples=10, colsample_bytree=0.3, reg_lambda=5, random_state=model seed; структура, групповая доля, внешние признаки |
| Структура | Morgan counts radius=2, 2048 координат; MACCS 167 координат; некорректные SMILES дают нулевую строку |
| `crossfit` | baseline с внутренним 5-fold кодированием групповой доли (min(5, n_rows)); shuffle=True, random_state=model seed |
| `descriptors` | crossfit плюс RDKit descriptors из vacv/features.py |
| `context` | только context, исходные настройки ExtraTrees |
| `context_full` | только context, min_samples_leaf=1, max_features=1.0 |
| `context_half` | только context, min_samples_leaf=1, max_features=0.5 |
| `context_leaf1` | только context, min_samples_leaf=1, max_features=sqrt |
| `kernel` | ridge alpha=0.3; K=Tanimoto(binary Morgan)^2 + 2*same_group + same_group*same_matched; group=(strain,Culture); центрирование меток по train prior, результат clip[0,1] |
| `kernel1` | то же, alpha=1.0 |
| CatBoost общий | 1000 итераций, learning_rate=0.035, l2_leaf_reg=5, Logloss, Ordered boosting, thread_count=4, random_seed=model seed; категории strain/Culture/group, числовые Title/внешние признаки |
| `catboost` / `catboost6` | depth=4 / 6; дополнительно Morgan counts, MACCS, RDKit descriptors |
| `catcontext` | depth=4, без молекулярных признаков |

Неуказанные настройки библиотек соответствуют версиям из `environment.json` и
`uv.lock`. `blend` реализован как равная смесь baseline/context/kernel, но в данном
этапе не оценивался и не выбирался. Точные определения — `vacv/inference.py`,
`vacv/pipeline.py`, `vacv/kernel.py`, `vacv/boosted.py`.

## Воспроизведение

Рабочая ветка: `codex/model-validation-improvements`, исходный коммит
`533c9c8bc26b6edfd2d30fad8edcf9ec321459cf`. Для воспроизведения нужен коммит,
содержащий этот отчёт, его `uv.lock`, указанные входные файлы и сохранённый снимок
внешних признаков. Окружение: macOS Apple Silicon, Python 3.13.14; версии основных
библиотек — `environment.json`.

```bash
uv sync --frozen
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/sklearn/.dylibs"
export PYTHONPATH=.
uv run python scripts/cv.py --profiles baseline crossfit descriptors context --seeds 1 --repeats 3 --out outputs/screen
uv run python scripts/cv.py --profiles context_full context_half context_leaf1 --seeds 1 --repeats 3 --out outputs/context-screen
uv run python scripts/cv.py --profiles kernel kernel1 --seeds 1 --repeats 3 --out outputs/kernel-screen
uv run python scripts/cv.py --profiles catboost catboost6 catcontext --seeds 1 --repeats 3 --out outputs/catboost-screen
uv run python scripts/cv.py --profiles baseline context_half hybrid --seeds 3 --repeats 3 --start-seed 3 --out outputs/confirmation
uv run python scripts/cv.py --profiles baseline hybrid --seeds 7 --repeats 3 --start-seed 3 --out outputs/final-validation
uv run python scripts/submit.py --profile hybrid --seeds 7 --out outputs/submission-hybrid.csv
uv run python scripts/submit.py --profile baseline --seeds 7 --out outputs/submission-baseline.csv
uv run pytest -q
git diff --check
```

Путь OpenMP выше соответствует фактически использованному Python 3.13. В README
приведён вариант определения пути для другого Python. Команды не меняют системную
установку OpenMP. Построчные OOF остаются в игнорируемом `outputs/`; SHA256 каждого
использованного OOF входит в `metrics.json`. Сырые данные и локальное окружение
в этот коммит не включаются.

## Проверки и ограничения

- `DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/sklearn/.dylibs" PYTHONPATH=. uv run pytest -q` — 108 passed.
- `git diff --check` — пройден.
- Добавлены регрессионные проверки точного sentinel, single-class, invalid SMILES,
  изоляции внутреннего кодирования, shared inference/CV, удаления запрещённых
  колонок и валидации submission. Литературная подстановка проверена в shared path.
- Все опубликованные ниже значения получены локально; новый leaderboard score
  неизвестен. Повторные splits и высокая доля общего контекста не гарантируют
  перенос улучшения на публичную или приватную часть Kaggle.
- Для небольшого датасета MSE сильно зависит от единичных больших целей. Значение
  36066.667 намеренно оставлено в основной метрике; его вклад указан отдельно ниже.
- Внешние признаки и правила литературного сопоставления уже существовали до
  этого этапа и могли разрабатываться с учётом всех train-строк. CV не оценивает
  независимость этого исторического выбора признаков.

## Измеренные результаты

<!-- GENERATED_RESULTS -->

Финальная проверка (семь model seeds, MSE; меньше — лучше):

| Разбиение | Baseline | Hybrid | Снижение MSE |
|---|---:|---:|---:|
| random-3 | 9,734,828 | 8,803,503 | 9.57% |
| random-4 | 9,781,174 | 9,049,160 | 7.48% |
| random-5 | 9,783,440 | 8,977,600 | 8.24% |
| scaffold | 14,254,991 | 11,965,575 | 16.06% |

Средний random MSE: baseline **9,766,480**, hybrid **8,943,421**; снижение **8.43%**.

Все серии экспериментов (MSE в миллионах; std — популяционное стандартное
отклонение трёх split-метрик, не доверительный интервал):

| Серия | Профиль | Model seeds | Random mean ± std | Scaffold |
|---|---|---:|---:|---:|
| screen | baseline | 1 | 10.2016 ± 0.2755 | 14.3013 |
| screen | crossfit | 1 | 10.6892 ± 0.3141 | 14.3409 |
| screen | descriptors | 1 | 10.8980 ± 0.3100 | 14.7137 |
| screen | context | 1 | 10.3532 ± 0.2036 | 13.1197 |
| context-screen | context_full | 1 | 10.0580 ± 0.3113 | 12.9341 |
| context-screen | context_half | 1 | 9.9582 ± 0.2695 | 12.5251 |
| context-screen | context_leaf1 | 1 | 10.1779 ± 0.2294 | 12.7775 |
| kernel-screen | kernel | 1 | 10.1188 ± 0.1800 | 12.5775 |
| kernel-screen | kernel1 | 1 | 10.3989 ± 0.2016 | 12.9549 |
| catboost-screen | catboost | 1 | 12.3121 ± 0.3259 | 15.6446 |
| catboost-screen | catboost6 | 1 | 12.1757 ± 0.5793 | 15.4576 |
| catboost-screen | catcontext | 1 | 11.1107 ± 0.3636 | 14.0436 |
| confirmation | baseline | 3 | 9.7711 ± 0.0318 | 14.2475 |
| confirmation | context_half | 3 | 9.1171 ± 0.0789 | 12.5056 |
| confirmation | hybrid | 3 | 8.9429 ± 0.1001 | 11.9695 |
| final-validation | baseline | 7 | 9.7665 ± 0.0224 | 14.2550 |
| final-validation | hybrid | 7 | 8.9434 ± 0.1032 | 11.9656 |
| hybrid-screen-derived | hybrid | 1 | 9.5694 ± 0.2048 | 11.9779 |

`hybrid-screen-derived` вычислен из среднего построчных OOF `context_half` и
`kernel`, после проверки совпадения ID, target и fold. Confirmation/final
получены прямым вызовом общего `predict_ensemble`. Все 72 набора метрик,
включая AUC/Brier/MSE до подстановки, сохранены без округления в `metrics.json`.

Вклад единственного наблюдения SI=36066.667 в полную MSE:

| Разбиение | Профиль | Прогноз | Вклад в MSE |
|---|---|---:|---:|
| random-3 | baseline | 285.72 | 4,000,863 |
| random-3 | hybrid | 153.65 | 4,030,452 |
| random-4 | baseline | 556.24 | 3,940,596 |
| random-4 | hybrid | 270.53 | 4,004,262 |
| random-5 | baseline | 280.80 | 4,001,964 |
| random-5 | hybrid | 132.66 | 4,035,166 |
| scaffold | baseline | 210.86 | 4,017,623 |
| scaffold | hybrid | 133.02 | 4,035,085 |

Оба submission содержат 597 строк; повторно проверены порядок и множество ID,
названия колонок, конечность, неотрицательность, уникальность ID и SHA256.
Готовые файлы: [submission-hybrid.csv](submission-hybrid.csv),
[submission-baseline.csv](submission-baseline.csv); рядом JSON с profile/seeds/rows/SHA256.
Ни push в удалённый репозиторий, ни отправка на Kaggle в рамках этой проверки не выполнялись.
