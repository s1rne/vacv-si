# Промежуточный лучший чекпоинт — 2026-11-07

Это промежуточная сборка после независимых экспериментов основного агента, Astra и ресерча Sol. Не финальная версия проекта и не подтверждённое место на Kaggle.

## Что собрано

1. **Hybrid:** ExtraTrees по разрешённому контексту/внешним признакам и kernel ridge по Morgan/Tanimoto, источнику и статусу литературного совпадения. Равные веса, семь seed леса.
2. **Графовая поправка Astra:** внутри каждого outer-train четыре inner-фолда строят собственные OOF-ошибки вероятности sentinel. Они распространяются по молекулярному графу; поправка ослабляется фиксированным `0.25 × 4p(1−p)`. Тестовые строки участвуют только как неразмеченная геометрия.
3. **Предобученная химия Astra:** frozen CheMeleon (2048 признаков), PCA32, обученная только на текущем train, и TabICLv2 с двумя estimators. Ни Chemprop, ни TabICL не получают датасет через внешний API.
4. **Итоговые веса:** 75% вероятности графового hybrid + 25% вероятности CheMeleon/TabICL. Эти веса перенесены из фиксированной смеси CheMeleon, новый поиск коэффициентов не выполнялся.
5. **Мой postprocess:** один раз применяем условное среднее `p × 10000 + (1−p) × public_SI` к строкам с литературным SI. Для остальных используем среднее активных, рассчитанное только на train. Это сохраняет вероятность sentinel при неоднозначном внешнем совпадении.

`SI == 10000` определяется точным равенством. Реальный SI=36066.667 не удалён. Конкурсные CC50_uM, IC50_uM_VACV, их пропуски и производные в моделях не используются.

## Сравнение с началом

MSE на исходной шкале, меньше лучше. Random — среднее одинаковых пятифолдовых разбиений с seed 3, 4, 5; семь seed hybrid в обеих сравниваемых версиях. Исходный baseline воспроизводит первоначальные 0.3 ExtraTrees context + 0.7 LightGBM/group-rate и жёсткую литературную подстановку.

| Этап | Random MSE | Scaffold MSE |
|---|---:|---:|
| Исходная модель | 9,766,480 | 14,254,991 |
| Мой hybrid | 8,943,421 | 11,965,575 |
| Hybrid + вероятностная подстановка | 8,825,766 | 11,969,000 |
| Astra graph + soft | 8,706,572 | 11,867,767 |
| Astra CheMeleon/TabICL + soft | 8,707,875 | 11,882,621 |
| **Общий чекпоинт** | **8,634,637** | **11,850,296** |

**Изменение к исходной модели: −11.59% random MSE и −16.87% scaffold MSE.** На каждом из трёх random-разбиений комбинация лучше исходной модели и графового компонента. На random-5 отдельная CheMeleon-смесь немного лучше комбинации; выбирается лучший средний результат:

| Разбиение | Исходная модель | Чекпоинт |
|---|---:|---:|
| random-3 | 9,734,828 | 8,342,878 |
| random-4 | 9,781,174 | 8,721,742 |
| random-5 | 9,783,440 | 8,839,291 |

**Граница доказательства:** отдельные графовый и CheMeleon-кандидаты заморожены до их confirmation на splits 3–5. Их итоговая комбинация оценена **post-hoc на сохранённых OOF тех же разбиений**, с проверкой одинаковых row IDs/folds и исходных hybrid probabilities. Это не новый независимый holdout для комбинации. Полный inference на всех 320 обучающих строках затем заново выполнен объединённым кодом; исходные графовые и химические компоненты воспроизведены с нулевым расхождением. Полную CV объединённым entrypoint ещё не перезапускали; он добавлен для следующей проверки. Частичный повторный выбор моделей по одним 320 строкам оставляет selection bias. Новый public/private leaderboard score неизвестен.

Scaffold — отдельный стресс-тест на новые химические серии, не гарантированная нижняя граница leaderboard. Трансдуктивность графа нужно учитывать при сверке правил соревнования; открытая страница rules не была доступна. Никакой Kaggle submission не отправлен.

## Что не вошло

CatBoost, cross-fitted group-rate вместо исходного обучения, простое добавление RDKit-дескрипторов, richer title features, SVC и raw regression, прямой harmonic graph, source-kernel logistic, TabICL без pretrained chemistry, а также иерархическая empirical-Bayes модель источников проверены и не дали достаточного выигрыша для этой сборки. Отрицательные результаты не удалены и сохранены в собственных ветках.

## Готовые файлы

- `submission.csv`: настоящий итоговый CSV, 597 строк в порядке sample_submit.csv.
- `submission-components.csv`: вероятности hybrid, graph и chemical, итоговые raw/final predictions.
- `submission.json`: конфигурация, SHA256 CSV/components/кода, embedding hash.
- `metrics.json`, `oof-*.csv`: post-hoc сравнение на строго выровненных фолдах.
- `verification.json`: формат, конечность, неотрицательность, воспроизведение компонентов.

SHA256 submission: `acffbc276491daf001d369aed571e6c699ea6ef5e03aeafca043e8f805126655`.

## Воспроизведение

Исходные конкурсные файлы должны лежать в data/raw/kaggle. Для точного воспроизведения сохранены **два окружения**: encoder использует Chemprop/RDKit2026.3.6 из-за зависимости cuik-molmaker; model heads и все сравнения используют исходный RDKit2026.9.1. Не заменять RDKit основного окружения версией encoder.

```bash
uv sync --frozen
uv pip install --python .venv/bin/python -r experiments/results/2026-11-07-chemeleon/heads-requirements.txt
uv venv .venv-encoder --python 3.13
uv pip install --python .venv-encoder/bin/python -r experiments/results/2026-11-07-chemeleon/encoder-requirements.txt
mkdir -p data/raw/models
curl --fail --location --output data/raw/models/chemeleon_mp.pt https://zenodo.org/api/records/15460715/files/chemeleon_mp.pt/content
PYTHONPATH=. .venv-encoder/bin/python experiments/chemeleon_features.py
.venv/bin/python scripts/fetch_tabicl.py
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/sklearn/.dylibs"
PYTHONPATH=. .venv/bin/python scripts/submit_checkpoint.py
PYTHONPATH=. .venv/bin/python -m pytest -q
```

CheMeleon encoder проверяет MD5, TabICL downloader — закреплённые revision/размер/SHA256. Готовый локальный embedding cache также сверяется по SHA256. При наличии проверенного cache и checkpoint первые шаги подготовки не нужны. На других ОС настройка DYLD_LIBRARY_PATH не требуется.

Для повторного полного CV или новых разбиений (это команды для следующего этапа, не уже проведённые новые испытания):

```bash
PYTHONPATH=. .venv/bin/python scripts/cv.py --profiles baseline --seeds 7 --start-seed 3 --out outputs/replay-baseline
PYTHONPATH=. .venv/bin/python scripts/cv_checkpoint.py --splits 3 4 5 scaffold --out outputs/replay-checkpoint
# Следующий заранее фиксируемый эксперимент может использовать splits 6 7 8.
```

Реально выполнены: полный submit_checkpoint.py, template/finite/index/SHA проверка; нулевое расхождение графового компонента и восстановленных float32 вероятностей chemical с независимыми экспортами; **146 tests passed**; CLI-help cv_checkpoint.py. Модель не деплоилась и не отправлялась в Kaggle.

## Происхождение работы и ветки

| Ветка | Исходный коммит | Содержимое |
|---|---|---|
| codex/model-validation-improvements | 2958cdb | baseline, hybrid, точная CV, результаты |
| codex/novel-models | 3b7fc3a | граф/source-kernel/TabICL эксперименты, soft lookup |
| codex/astra-alternative-models | 0b59bd6 | title/SVC/raw-regression, отрицательные и контрольные результаты |
| codex/astra-hierarchical-models | 6b97e45 | Bayesian source/chemical-series модели, отрицательный результат |
| codex/astra-graph-correction | 20609cf | nested residual graph, confidence gate, final7 |
| codex/astra-pretrained-embeddings | 34e3927 | frozen CheMeleon, TabICL, final7 |
| codex/research-winning-methods | 18b48d3 | ресерч Sol с первичными источниками |

Сборка codex/best-model-checkpoint содержит рабочие компоненты graph и embeddings, мои изменения и эту сводку. История отрицательных вариантов остаётся в отдельных локальных ветках. Push не выполнялся.
