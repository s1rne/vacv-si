# Дополнительные методы и вероятностная подстановка литературы

Дата: 2026-11-07. База: `2958cdb`. Все оценки относятся к локальной CV; новые файлы не отправлены на Kaggle.

## Что проверено

- **Harmonic graph:** разреженный граф Morgan Tanimoto³ с сильными рёбрами внутри источника; решается система Лапласа с фиксированными метками только outer-train. В граф включены 597 неразмеченных тестовых объектов. Есть контроль без межгрупповых рёбер и вариант мягкого prior от публичной литературы.
- **Source kernel logistic:** общая химия, эффект источника и взаимодействие химии с источником; train-only разложение ядра и L2-логистика с C=3/10.
- **TabICLv2:** настоящий локальный pretrained classifier на 16 RDKit дескрипторах, пяти признаках названия, публичных признаках и двух категориях; два ensemble estimators. На random-0 и scaffold хуже, дальнейший подбор не выполнялся.
- **hybrid_soft:** сохраняет вероятность sentinel даже при наличии литературного SI: `p * 10000 + (1-p) * public_SI`. На unmatched строках прогноз hybrid не меняется. Это фиксированная формула условного среднего, без подбора порога или веса.

## Результаты исследовательских моделей

Во всех строках этого раздела random означает среднее разбиений 0–2; TabICL имеет только random-0. Литературная подстановка здесь исходная, жёсткая. Нижний MSE лучше.

| Профиль | Random MSE | Scaffold MSE |
|---|---:|---:|
| graph | 10,142,363 | 11,783,767 |
| graph_local | 10,129,230 | 11,649,438 |
| graph_evidence | 10,115,326 | 11,720,401 |
| graph_smooth | 10,957,421 | 13,819,928 |
| source_logistic | 10,396,729 | 13,787,330 |
| source_logistic10 | 10,111,660 | 13,614,739 |
| tabicl | 10,823,716 | 15,701,390 |

Для сравнения hybrid на тех же screening splits: random **9,569,396**, scaffold **11,977,887**. Ни один отдельный исследовательский метод не заменил hybrid. Граф улучшил scaffold, но оказался хуже на random; простая 25% примесь graph_local дала разнонаправленные изменения по random folds и не продвигалась.

## Проверка вероятностной подстановки

Формула `hybrid_soft` сначала проверена на screening 0–2: 9,428,580 / 9,722,363 / 9,214,673 против 9,514,975 / 9,843,005 / 9,350,207 у hard lookup. Формула затем без изменений применена к сохранённым OOF финальной семисидовой модели на splits 3–5. Это точное переиспользование тех же model predictions/probabilities, а не повторное обучение или замена фолдов.

| Разбиение | Hard lookup | Conditional lookup |
|---|---:|---:|
| random-3 | 8,803,503 | 8,656,653 |
| random-4 | 9,049,160 | 8,897,749 |
| random-5 | 8,977,600 | 8,922,895 |
| scaffold | 11,965,575 | 11,969,000 |

**Среднее random: 8,825,766** — на 1.32% ниже hybrid hard и на **9.63% ниже исходного baseline 9,766,480**. На scaffold soft хуже hard на 0.029%, то есть практически одинаково; улучшение не объявляется универсальным.

## Воспроизведение

Рабочая папка — корень репозитория. На macOS использована процессная настройка OpenMP из README. Дополнительные зависимости находятся только в отдельной `.venv`; основной lockfile не менялся.

```bash
uv sync --frozen
uv pip install --python .venv/bin/python -r experiments/requirements-novel.txt
export PYTHONPATH=.
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/sklearn/.dylibs"
.venv/bin/python experiments/graph_methods.py --profiles graph graph_local graph_evidence graph_smooth --out outputs/reproduce-graphs
.venv/bin/python experiments/graph_methods.py --profiles source_logistic source_logistic10 --out outputs/reproduce-source
.venv/bin/python scripts/fetch_tabicl.py
.venv/bin/python experiments/graph_methods.py --profiles tabicl --repeats 1 --out outputs/reproduce-tabicl
.venv/bin/python scripts/cv.py --profiles baseline hybrid_soft --seeds 7 --start-seed 3 --out outputs/reproduce-soft
.venv/bin/python scripts/submit.py --profile hybrid_soft --seeds 7 --out outputs/submission-soft.csv
.venv/bin/python -m pytest -q
```

Первоначально source/graph/TabICL запускались ровно такими же командами с каталогами outputs/graphs, outputs/source-kernel, outputs/tabicl-pilot. Финальные soft CV значения вычислены из outputs/final-validation/*-hybrid.csv базового прогона с проверкой одинакового порядка целевых значений. Приведённая команда CV воспроизводит тот же расчёт с повторным обучением.

## Артефакты и ограничения

- `metrics.json`: 30 результатов моделей; `soft-lookup-screen.csv`: ещё четыре результата фиксированного postprocess. OOF генерируются в ignored outputs. Никакие train/test CSV в этот коммит не добавлены.
- `submission-soft.csv` и JSON: 597 строк, профиль hybrid_soft, семь seed; CSV сгенерирован настоящим final inference. Нужны только разрешённые входные признаки.
- `environment.json`: фактические версии; TabICL checkpoint revision/SHA зафиксированы в downloader, веса остаются ignored. Inference CPU, никакой отправки SMILES/датасета в облако.
- `SI VACV == 10000` проверяется точным равенством; значение 36066.667 не удалялось, входит в активную часть и полный MSE.
- Все методы используют loader, удаляющий конкурсные CC50_uM/IC50_uM_VACV. Валидационные и тестовые target/assay поля игнорируются также явными whitelist внутри графа и foundation features.
- Внешние источники и их признаки унаследованы от исходного репозитория; новый сбор не выполнялся. Их исторический выбор по всему train остаётся ограничением оценки.
- Повторные разбиения тех же 320 строк не являются независимым финальным тестом; дальнейший отбор разных методов тоже вносит selection bias. Лидерборд и private score не измерены.
- Графы являются трансдуктивными и используют только неразмеченную геометрию теста. Возможность такого режима следует сверять с правилами; официальный rules URL не удалось прочитать. Выбранный hybrid_soft — индуктивный, без этой зависимости.
- TabICL и source logistic дали отрицательные результаты. Их наличие в коде не означает рекомендацию заменить рабочую модель.

Идеи foundation моделей сопоставлены с [разбором OpenADMET](https://openadmet.ghost.io/are-tabular-foundation-models-all-you-need/) и [TabICLv2](https://huggingface.co/jingang/TabICL). Независимые направления других веток документируются отдельно; в этом отчёте их будущие результаты не приписываются текущему коду.

Проверки на момент сохранения: полный suite **118 passed**, pinned checkpoint SHA256 проверен. Финальная инспекция проводится перед коммитом.
