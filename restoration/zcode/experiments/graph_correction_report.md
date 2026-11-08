# Графовая коррекция ошибок сильного hybrid

2026-11-07; ветка `codex/astra-graph-correction`, исходная база `2958cdb`.
Изолированная рабочая копия; исходный production-профиль hybrid не изменён.

Итог: зафиксированная confidence-gated коррекция с семью seed и мягкой литературной
подстановкой дала средний confirmation MSE **8.706572M**, против **8.825766M**
у hybrid с той же подстановкой (−1.351%). По остовам 11.867767M против 11.969000M.
Это улучшение CV, без проверки на Kaggle leaderboard.

## Идея и границы

[Correct and Smooth, Huang et al., 2020](https://arxiv.org/abs/2010.13993)
предлагает распространять коррелированные ошибки базовой модели по графу.
Здесь адаптирована только коррекция с фиксированными граничными остатками;
это не воспроизведение полного C&S pipeline и не перенос его benchmark-результатов.

Базовая модель — существующий hybrid: 50% context ExtraTrees + 50% kernel ridge.
Графовая геометрия передана основной веткой в `vacv/transductive.py`:
Morgan Tanimoto³, k=6, множитель cross-group 0.05 и слабые полные внутригрупповые
связи 0.1; группа — точная пара strain+Culture. В графе 917 узлов: outer train,
outer validation и разрешённые признаки 597 test-строк. Это transductive метод:
прогноз зависит от набора совместно предоставленных немеченых молекул.

Все данные соревнования проходят `dataset.load()`. Запрещённые CC50_uM,
IC50_uM_VACV и их missingness не используются. Дополнительно `safe_features`
удаляет target и любые посторонние колонки перед моделями. Граф использует только
SMILES и контекст. Публичные литературные признаки унаследованы от базы.
Метка неактивности — **ровно y==10000**; настоящий 36066.667 остаётся активным.

## Защита от утечки при коррекции

В каждом outer train отдельно выполняется KFold(4, shuffle=True, random_state=2718).
Каждый inner hybrid обучается без метки соответствующей inner validation строки.
Остаток `r = (y==10000) - p_inner_oof` доступен только на outer train.
Внешние OOF других outer folds не используются: их обучающие наборы могли содержать
текущие outer validation labels. Внешние метки нужны исключительно для подсчёта метрик.

Обозначим размеченные узлы L, немеченые U. Коррекции:

- one-hop: `delta_u = W_ul r_l / (sum_l W_ul + 0.03)`; U–U связи удалены
  из того же общего графа, включая их вклад в степень;
- diffusion: `(D_uu + 0.03 I - W_uu) delta_u = W_ul r_l`, то есть фиксированные
  residual boundaries и нулевой prior коррекции на U; U–U связи сохраняются;
- probability: `clip(p_hybrid + beta*delta, 0, 1)`, фиксированные beta 0.25 и 0.5;
- SI: `p*10000 + (1-p)*mean_active`, где mean_active вычислен только по outer train.
  Затем применяется такая же литературная подстановка, как при полном обучении.

## Исследовательские результаты

MSE в миллионах, меньше лучше. Model seeds=1 означает один seed **0**, не seed=1.
Random split seeds 0–2; scaffold split — существующая детерминированная группировка
по Murcko. Каждая метрика учитывает все 320 строк, включая настоящий выброс.

| Вариант | Random 0 | Random 1 | Random 2 | Scaffold |
|---|---:|---:|---:|---:|
| Hybrid | 9.514975 | 9.843005 | 9.350207 | 11.977887 |
| One-hop, beta 0.25 | 9.657175 | 10.110774 | 9.391088 | 11.859014 |
| One-hop, beta 0.5 | 10.106099 | 10.532631 | 9.677688 | 11.923942 |
| Diffusion, beta 0.25 | 9.560527 | 9.832742 | 9.352680 | 11.909792 |
| Diffusion, beta 0.5 | 9.751325 | 9.923242 | 9.493896 | 11.982052 |

Исходная гипотеза не дала выигрыша на random CV: mean MSE diffusion beta 0.25
9.581983M против 9.569396M у hybrid (+0.13%). One-hop оказался хуже.

После просмотра этих результатов проверены **две post-hoc exploratory** корректировки
`clip(p + beta*4*p*(1-p)*delta_diffusion, 0, 1)`. Gate равен нулю при уверенном
прогнозе 0/1 и единице при p=0.5; сама формула и beta не обучались.

| Gate | Random 0 | Random 1 | Random 2 | Scaffold |
|---|---:|---:|---:|---:|
| beta 0.25 | 9.499572 | 9.802173 | 9.333624 | 11.877917 |
| beta 0.5 | 9.573809 | 9.847372 | 9.403882 | 11.871229 |

beta 0.25 дала mean 9.545123M (−0.254%), поэтому только она выбрана для новой
проверки. Выбор по этим данным не является независимым подтверждением.
План зафиксирован до confirmation в [graph_candidate_freeze.md](graph_candidate_freeze.md).

## Confirmation без изменения параметров

Новые random split seeds 3, 4, 5; один model seed 0; те же inner folds и геометрия.

| Вариант | Random 3 | Random 4 | Random 5 | Среднее |
|---|---:|---:|---:|---:|
| Hybrid | 8.815902 | 9.060875 | 9.003592 | 8.960123 |
| Gated diffusion, beta 0.25 | 8.602210 | 8.938484 | 9.011617 | 8.850770 |

Среднее улучшилось на 1.220%; два split лучше, третий хуже на 0.089%.
Параметры по confirmation не менялись. Повторные разбиения одних 320 объектов
проверяют устойчивость, но не заменяют независимый внешний тест.

## Финальный ансамбль: семь seed и вероятностная литература

Формула gate, beta и graph/inner настройки не менялись. После single-seed confirmation
проведён заранее записанный следующий шаг: seeds 0–6 **и внутри каждого inner fold,
и во внешнем hybrid**. Это проверка финального ансамбля на тех же confirmation splits,
не новая независимая выборка. Сравнение с soft выполнено по отдельному согласованному плану.

Soft-функция `vacv/postprocess.py:conditional_lookup` передана основной веткой
и скопирована побайтово без изменений. Для найденной литературы итог равен
`p*10000 + (1-p)*clip(literature_si, 0, 10000)`; для unmatched сохраняется raw SI.
Soft-формула не разрабатывалась и не подбиралась этим графовым экспериментом.

| Вариант, seeds 0–6 | Random 3 | Random 4 | Random 5 | Среднее random | Scaffold |
|---|---:|---:|---:|---:|---:|
| Hybrid hard | 8.803503 | 9.049160 | 8.977600 | 8.943421 | 11.965575 |
| Gated hard | 8.591750 | 8.924806 | 8.986741 | 8.834432 | 11.865706 |
| Hybrid soft | 8.656653 | 8.897749 | 8.922895 | 8.825766 | 11.969000 |
| Gated soft | 8.429371 | 8.765162 | 8.925181 | **8.706572** | **11.867767** |

Gated soft выигрывает у hybrid soft на split 3 и 4 (−2.63% и −1.49%);
split 5 хуже на 0.026%. Mean random −1.351%, scaffold −0.846%.
Относительно исходного hybrid hard mean улучшился на 2.648%.
Кандидат сохранён для возможной интеграции; обобщение на внешний leaderboard
не проверено, и небольшая величина выигрыша требует осторожного вывода.

Полное обучение использует 320 train-строк, внутренний crossfit 4×7 seeds,
outer hybrid 7 seeds и граф train+597 test features. Готовый **неотправленный** CSV:
`results/graph_submission/submission.csv`; режим gated diffusion beta 0.25 + soft.
597 строк, совпадение порядка с `sample_submit.csv`, конечность/неотрицательность
и SHA256 проверены. Диапазон 0.302632–10000.0. Контрольная сумма:
`a6b8b72144429414af2ef2d7050a2426978c070426e98a15467b672c68f7327f`.

## Проверки и воспроизведение

Python 3.13.14; существующая shared venv использовалась только для чтения.
На macOS каждой команде ниже предшествовал process-local префикс:

```bash
DYLD_LIBRARY_PATH=/Users/mike/Projects/vacv-si/.venv/lib/python3.13/site-packages/sklearn/.dylibs PYTHONPATH=. /Users/mike/Projects/vacv-si/.venv/bin/python
```

Аргументы выполненных команд:

```text
experiments/graph_correction.py
experiments/graph_correction_gated.py
experiments/graph_correction.py --candidate gated_diffusion_0.25 --splits 3 4 5 --seeds 1 --out results/graph_confirmation
experiments/graph_correction.py --candidate gated_diffusion_0.25 --splits 3 4 5 scaffold --seeds 7 --with-soft --out results/graph_final7
experiments/graph_correction.py --candidate gated_diffusion_0.25 --submit diffusion --beta 0.25 --seeds 7 --with-soft --out results/graph_submission
-m pytest -q
```

Для повторения в обычной среде использовать `PYTHONPATH=. uv run python` и те же
аргументы. Непустые каталоги защищены от перезаписи: новый `--out` либо явный `--overwrite`.
Результаты: `results/graph_screen`, `graph_gated`, `graph_confirmation`, `graph_final7`;
JSON содержит точные метрики, CSV — OOF probabilities/predictions, fold IDs и graph
deltas, **без исходных target values**. `config.json` сохраняет параметры запуска.

`python -m pytest -q`: **120 passed**. Тесты проверяют исключение своей метки
из inner prediction, отсутствие outer validation rows в обучении, whitelist,
аналитическое решение диффузии, разорванные компоненты, активный выброс,
confidence gate, литературу и защиту результатов от перезаписи.
Все **50** сохранённых MSE независимо пересчитаны из соответствующих OOF и совпали
с JSON. `git diff --check`: passed. Секреты/raw данные не включены;
push и Kaggle submit не выполнялись.
