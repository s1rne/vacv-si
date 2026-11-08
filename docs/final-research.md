# Хронология экспериментов и обоснование финальной модели

Этот документ сводит результаты исследований по предсказанию SI VACV и фиксирует, почему текущим подтверждённым решением остаётся Ridge-log pair-delta v4. Он дополняет, а не заменяет подробные протоколы и OOF-артефакты. Полный каталог из 54 каталогов экспериментов и результатов — [experiment-inventory.json](experiment-inventory.json).

## Как читать числа

Основной локальный показатель — полная raw-MSE по всем 320 строкам официального `train_v2`, без удаления, обрезки или замены максимальной метки 36 066,667. В повторяющихся случайных CV-разбиениях используются те же строки, поэтому seed-ы не являются независимыми выборками. Scaffold и identity GroupKFold — отдельные проверки переноса и устойчивости; сравнивать модели следует по парному контролю на тех же фолдах.

Исторические результаты относятся к разным версиям target и разным контурам проверки. До `train_v2` значение 10 000 использовалось как sentinel для неактивных соединений; официальная версия изменила 81 из 320 меток с 10 000 на 0. Поэтому первоначальные CV и публичные результаты для sentinel-моделей нельзя сравнивать с регрессией на текущих метках. Ранние Kaggle public MSE 18,146,452.74419 и видимый прежний лучший результат 14,633,838.88963 относятся к историческому этапу; они не являются baseline на текущей версии target. После перехода на v2 контекстно-структурная регрессия получила Kaggle Public MSE 2,038,068.29444. Замороженный pair-delta v4 затем улучшил этот подтверждённый Public результат на 16,09% — до 1,710,149.26680. Public score и локальный CV измеряют разные строки и напрямую не сопоставляются.

## Этапы

### 1. Раннее восстановление и исправление target

Первый clean-input этап сравнивал гибридные и графовые варианты на старой постановке. На random CV средние confirmation MSE были порядка 8,8–9,0 млн; scaffold MSE у выбранного sharpened hybrid составил 12,36 млн. Его Kaggle результат 18,15 млн оказался хуже прежнего видимого best 14,63 млн. Эти цифры относятся к прежней цели и не характеризуют качество на `train_v2`.

После получения официального `train_v2.csv` аудит обнаружил 81 изменение меток 10 000 → 0 при совпадающих допустимых признаках и неизменном тестовом файле. Старые модели, пересчитанные на новых метках, дали 22,87–25,07 млн MSE: старый sentinel-классификатор оптимизировал не ту задачу. Новая прямая регрессия по SI (все 320 меток сохранены) дала контекстно-структурному ET-log confirmation mean 4,241 млн на seeds 7/8, scaffold 4,263 млн. Kaggle Public MSE для v2-модели составил 2,038,068.29444.

В нынешней интеграции загрузчики и команды проверки учитывают текущую версию target; исторический sentinel replay требует явного `--allow-legacy-target`. Проверка устаревшего 10000-протокола вынесена в `vacv/target_version.py`. Текущую модель следует воспроизводить с официальным v2-входом, а не через legacy sentinel-пайплайн.

### 2. Регрессия, внешние данные и tail-экстраполяция (v2–v3)

На официальных метках сравнивались raw/log-регрессия, CheMeleon-представления, внешние presence-признаки и измерения. На фолдах 6–8 лучший базовый контекстно-структурный ET-log имел confirmation mean 4,241,095 и scaffold 4,262,617. CheMeleon-варианты и добавление внешних признаков дали небольшие изменения, но не устойчивое превосходство; внешнее измеренное обучение улучшило парный результат примерно на 0,45%, один confirmation при этом ухудшился.

Power-преобразование, выбранное на seed 9, дало 4,249,129 mean на confirmation 10/11 против 4,268,083 у log baseline, но не достигло локального порога 3 млн на каждом фолде. Попытка sqrt-Ridge для экстраполяции провалилась: MSE 1,673×10¹⁶ и 5,012×10¹² на seeds 16/17; нестабильность была связана с экстремальным значением признака Ipc и предсказаниями для обычных молекул. Варианты с поправкой условной дисперсии тоже ухудшили результаты.

Сборщик внешних данных сначала смешивал отдельные активности, измерительные endpoints и цензурированные границы. Аудит запретил считать агрегатные совпадения подтверждёнными метками. Исправленный source-pair маршрут разделил 47 точных пар и 123 интервальных наблюдения; интервальная регрессия на seeds 22/23 оставалась примерно на уровне 4,274 млн и предсказывала максимум лишь около 63–71. Условия assay для части записей известны не полностью, поэтому наличие внешней метки не означает доказанной полной протокольной эквивалентности.

### 3. Выбор v4 pair-delta

Проверялись product context/structure kernel, pair-delta Ridge/ExtraTrees и узкий single-cut MMP эксперт. Product kernel проиграл своему парному baseline на seeds 31/32: confirmation mean 4,225 млн против 4,010 млн. MMP покрывал только 13–16 строк на random фолдах и не покрывал большие SI или scaffold; разница с контролем была практически нулевой.

Pair-delta Ridge-log, отобранный на discovery seed 33, дал 3,132,248 на seed 34 против 3,778,835 у контроля, но на seed 35 проиграл: 4,111,629 против 4,061,465. Scaffold MSE — 4,253,853. На заранее выбранных дополнительных seeds 13/14/39/40 модель выиграла у парного контроля на каждом фолде: среднее 3,455,090 против 4,045,481. Совместно шесть недискавери-разбиений 13/14/34/35/39/40 дали 3,510,706 против 4,003,704 (улучшение 12,31%), 5 побед из 6. Повторное использование одних 320 объектов и чувствительность к тому, куда попадает максимальная метка, не позволяют трактовать это как независимую проверку.

Замороженная модель была отправлена на Kaggle; её Public MSE 1,710,149.26680 на 16,09% ниже предыдущего подтверждённого v2 результата 2,038,068.29444. Это и есть финальный подтверждённый результат соревнования в материалах репозитория. Он не доказывает универсальную точность на новой выборке.

**Алгоритм v4.** Для каждого внешнего train-фолда строятся Morgan radius-2 binary fingerprints и Tanimoto-сходства. Обучающие молекулярные пары выбираются только из этого train: одинаковый нормализованный контекст strain/culture, сходство не ниже 0,5 и top-8 соседей хотя бы с одной стороны. Для пары используются оба направления. Ridge без intercept (`alpha=10`) обучается на разностях fingerprint и разностях `log1p(SI)`. При предсказании берутся до трёх train-якорей в совпадающем контексте с Tanimoto не ниже 0,7; Ridge оценивает разность для query относительно каждого якоря, а антисимметричная обёртка усредняет оценки для `d` и `-d` с обращением знака. Изменения в log-шкале прибавляются к `log1p` якорей, затем инвертируются; оценки якорей агрегируются весами `similarity^8`, результат снизу ограничивается нулём. Если нет подходящих якорей или парного покрытия, используется исходный структурный log-KNN fallback (без требования совпадения контекста). Верхнего clipping нет. Все обучаемые разности и якорные target-значения принадлежат внешнему train-фолду.

### 4. Проверки v5 и v6: удержание incumbent

На общих confirmation folds 42–45 ни одна из проверок v5 не заменила v4: Ridge alpha 100, pair bagging, radius-2+3 fingerprints и фиксированный ансамбль дали mean MSE 3,781; 3,696; 4,136 и 3,812 млн против 3,681 млн у v4. Bagging выиграл отдельный seed 44 (2,962 млн), но проиграл на других фолдах. Tail weighting на seeds 47–50 проиграл все четыре подтверждения (3,905 млн против 3,252 млн); более мягкий anchor gate также проиграл все четыре. Identity GroupKFold5 для v4 дал 3,280 млн.

В v6 сравнивались нелинейный Tanimoto pair kernel, query-local Ridge, внешние пары, background-aware ExtraTrees и смещение log-target. На общих seeds 52–55 pair kernel получил 4,117 млн, local blend 3,755 млн, external cross-pair 3,690 млн — каждый вариант проиграл v4 по mean и не выиграл ни одного confirmation. Background trees на seeds 57–60 дали 3,921 млн против 3,411 млн; offset-10 — 3,936 млн против 3,411 млн. Малые offsets были особенно нестабильны: на discovery offset 0,01 дал 133,099 млн, offset 0,1 — 7,987 млн. Ни один вариант не продвигался.

GLM rebuild4 заменил прежние агрегаты записями, собранными из record-pair provenance. На фиксированных folds 52–55 обновлённые 43 внешние записи изменили cross-pair mean лишь с 3,690,391 до 3,690,145; v4 оставался лучше с 3,595,103. Четыре исключённые старые записи не были доказаны ошибочными, а NIOCH-14 является альтернативным измерением, не «исправлением» прежней цифры. См. `docs/glm-v4-audit.md`, `docs/glm-rebuild4-audit.md` и отчёт `experiments/results/glm-rebuild4-ablation/report.md`.

### 5. Nested selection и backend comparison (v8–v9)

V8 заранее зафиксировал identity GroupKFold5 и две перемешанные identity-повторности, отдельный identity/scaffold стресс-контур, три внутренних фолда в каждом outer train и подбор одиночной модели/весов ансамбля только по inner OOF. Все варианты сравнивались с неизменённым v4 на полных 320 метках. На primary identity blended outer MSE составил 2,920 млн против 3,280 млн у v4 (−10,99%); одиночная inner-selected модель дала 2,301 млн, но это вторичный endpoint. На повторностях blend проиграл: 4,321 против 4,312 млн (+0,20%) и 3,976 против 3,940 млн (+0,90%). Предзаданный критерий потребовал улучшения primary и обеих повторностей без ухудшения структурного стресса, поэтому не прошёл. V4 остался принятым; полного deployable ансамбля этот validation-only эксперимент не обучал.

V9 добавил ExtraTrees, CatBoost и Kernel Ridge на raw/log target, с отдельной external-data ablation. В среднем по трём identity-схемам у v4 было 3,844 млн; выбранный core ensemble получил 4,309 млн (+12,10%). Все backend-кандидаты в таблице проигрывали v4 по групповому среднему. Внешние данные сдвинули ExtraTrees-log group mean с 4,254,633 до 4,254,080 (−0,013%), при этом ухудшили один repeat; это не устойчивый выигрыш и не подтверждение assay-совместимости.

### 6. Chirality и калибровка (v10)

V10 использовал те же материализованные V8 outer/inner фолды и неизменённые все 320 target. Проверялись chiral Morgan-представление, скалярная калибровка, prediction cap 5000 и их фиксированные сочетания. Nested scalar калибровка ухудшила identity mean с 3,844 до 4,209 млн. Chiral вариант дал 4,263 млн, chiral+scalar — 4,270 млн; cap 5000 не изменил outer predictions. Chirality улучшила диагностику без максимальной строки, но full MSE ухудшила в двух из трёх identity схем. Ни один вариант не прошёл заданный robustness критерий; кандидаты не продвигались и новой отправки не было.

## Итог и границы вывода

Единственная модель, подтверждённая внешним Kaggle score как улучшение предыдущего актуального результата, — frozen v4 pair-delta с Public MSE 1,710,149.26680 и относительным улучшением 16,09% против v2 Public 2,038,068.29444. После неё локальные исследования V5–V10 не дали кандидата, который прошёл бы заранее установленную проверку устойчивости; локальная первичная победа, отдельный seed или улучшение метрики без максимальной строки не меняли это решение.

Не следует объединять старые sentinel scores с v2 scores, напрямую приравнивать CV к Kaggle Public, считать repeated folds независимыми данными или интерпретировать внешний record match как гарантию полного assay-протокола. Инвентаризация сохранённых артефактов, включая неудачные, исторические и только частично документированные каталоги, доступна в [experiment-inventory.json](experiment-inventory.json); само наличие CSV или submission-файла не означает успешную валидацию или фактическую отправку.


## Проверка итоговой интеграции main

В исходном main README направлял на исторический hybrid, а загрузчик переводил нулевые targets обратно в 10000. При интеграции перенесены regression-v2 загрузчик с whitelist полей и guard для явного legacy replay; устаревший тест заменён проверкой сохранения официального нуля. Финальный путь запуска теперь `scripts/reproduce_final.py`; исторические команды сохранены для исследовательского воспроизведения.

Полный inference в итоговом рабочем дереве: 320 train, 597 test, 1004 направленные обучающие пары, 277 test-строк с pair expert. Все значения и байты CSV совпали с исторической отправкой, maxAbs=0, SHA256 `cbd1b6c1e7ebb858c9ca9bda38dbc24392c7ef81271228ef4edce96edd87ab60`. Это проверка воспроизводимости прогноза; скрытые Kaggle targets локально недоступны. Машиночитаемые хеши кода, входов, версии окружения и проверка — [final-reproduction-verification.json](final-reproduction-verification.json).

`PYTHONPATH=. python -m pytest -q`: 396 passed в существующем Python 3.13.14 окружении. Для двух исторических тестов восстановления дополнительно подключён локальный старый `data/raw/kaggle`; для финального inference нужны только санитизированные `data/raw_v2`. Чистая установка `uv sync` в новом окружении в этой интеграции не проверялась. Исторические CSV сохранены неизменными, включая пробелы Culture, чтобы не нарушить контрольные суммы.

## Дополнительные направления и незавершённые материалы

До reset цели исследовались альтернативные деревья/ядра, иерархические модели, коррекция по графу сходства и предобученные CheMeleon/PCA/TabICL. Их чекпоинты и отчёты относятся к исходному target/CV и не доказывают преимущества над текущим v4. На v2 также сохранены Tweedie, hurdle, TabICL, embedding Ridge/ET, локальные KRR/KNN и Luna-ансамбли/устойчивая экстраполяция: они не прошли критерии отправки или не заменили v4; ссылки на все сохранённые материалы перечислены ниже. Исторический README и исходные подробные отчёты сохранены, а не переименованы в результаты финальной модели.

В recovery рабочем дереве присутствовали незакоммиченные V7 nested embeddings и Terra-материалы. Они не включены в main как завершённые проверенные результаты. Найденные V7 метрики были смешанными: 4,140 vs v4 4,146 млн на seed61; 4,172 vs 3,926 млн на seed62; scaffold 4,269 vs 4,254 млн. Это справочная фиксация незавершённого направления, не основание продвижения модели. Частичные OOF-файлы других агентов также не объявляются завершёнными запусками.

## Отдельный архив восстановления Z-Code

По запросу команды сохранён [снимок восстановления полной базы из интернет-источников](../restoration/zcode/README.md) из `zcode/max-lb`: таблицы и код. Проверено наличие материалов. Правильность, полнота восстановления и архивные методы в этой задаче повторно не проверялись. Эти данные не подключены к финальному v4.

## Указатель всех сохранённых запусков

Каждый каталог ниже является самостоятельным историческим контуром. Отсутствие отдельного report означает, что вывод нужно читать в метриках/manifest, а не считать эксперимент успешным. Полный список файлов доказательств и их SHA256 находится в JSON-инвентаризации.

| Каталог | Первичное описание / метрики |
|---|---|
| [2026-11-07](../experiments/results/2026-11-07) | [доказательства](../experiments/results/2026-11-07/report.md) |
| [2026-11-07-chemeleon](../experiments/results/2026-11-07-chemeleon) | [доказательства](../experiments/results/2026-11-07-chemeleon/report.md) |
| [2026-11-07-novel](../experiments/results/2026-11-07-novel) | [доказательства](../experiments/results/2026-11-07-novel/report.md) |
| [checkpoint-2026-11-07](../experiments/results/checkpoint-2026-11-07) | [доказательства](../experiments/results/checkpoint-2026-11-07/report.md) |
| [glm-rebuild4-ablation](../experiments/results/glm-rebuild4-ablation) | [доказательства](../experiments/results/glm-rebuild4-ablation/report.md) |
| [glm-rebuild4-audit](../experiments/results/glm-rebuild4-audit) | CSV / конфигурация в каталоге |
| [glm-v4-audit](../experiments/results/glm-v4-audit) | CSV / конфигурация в каталоге |
| [luna-v2-ensembles](../experiments/results/luna-v2-ensembles) | [доказательства](../experiments/results/luna-v2-ensembles/README.md) |
| [luna-v2-external-models](../experiments/results/luna-v2-external-models) | [доказательства](../experiments/results/luna-v2-external-models/report.md) |
| [luna-v2-stable-extrapolation](../experiments/results/luna-v2-stable-extrapolation) | [доказательства](../experiments/results/luna-v2-stable-extrapolation/README.md) |
| [recovery-candidates](../experiments/results/recovery-candidates) | [доказательства](../experiments/results/recovery-candidates/report.md) |
| [recovery-clean](../experiments/results/recovery-clean) | [доказательства](../experiments/results/recovery-clean/report.md) |
| [restore-baseline](../experiments/results/restore-baseline) | [доказательства](../experiments/results/restore-baseline/report.md) |
| [target-v2](../experiments/results/target-v2) | [доказательства](../experiments/results/target-v2/report.md) |
| [v10-spec-validation](../experiments/results/v10-spec-validation) | [доказательства](../experiments/results/v10-spec-validation/metrics.json) |
| [v2-analog](../experiments/results/v2-analog) | [доказательства](../experiments/results/v2-analog/README.md) |
| [v2-context-analog](../experiments/results/v2-context-analog) | [доказательства](../experiments/results/v2-context-analog/README.md) |
| [v2-embeddings](../experiments/results/v2-embeddings) | [доказательства](../experiments/results/v2-embeddings/report.md) |
| [v2-external](../experiments/results/v2-external) | [доказательства](../experiments/results/v2-external/metrics.json) |
| [v2-extrapolation](../experiments/results/v2-extrapolation) | [доказательства](../experiments/results/v2-extrapolation/README.md) |
| [v2-hurdle](../experiments/results/v2-hurdle) | [доказательства](../experiments/results/v2-hurdle/README.md) |
| [v2-power](../experiments/results/v2-power) | [доказательства](../experiments/results/v2-power/README.md) |
| [v2-tabicl](../experiments/results/v2-tabicl) | [доказательства](../experiments/results/v2-tabicl/README.md) |
| [v2-tweedie](../experiments/results/v2-tweedie) | [доказательства](../experiments/results/v2-tweedie/README.md) |
| [v3-distributional](../experiments/results/v3-distributional) | [доказательства](../experiments/results/v3-distributional/README.md) |
| [v3-interval-scales](../experiments/results/v3-interval-scales) | [доказательства](../experiments/results/v3-interval-scales/README.md) |
| [v3-intervals](../experiments/results/v3-intervals) | [доказательства](../experiments/results/v3-intervals/README.md) |
| [v3-source-pairs](../experiments/results/v3-source-pairs) | [доказательства](../experiments/results/v3-source-pairs/README.md) |
| [v4-context-kernel](../experiments/results/v4-context-kernel) | [доказательства](../experiments/results/v4-context-kernel/README.md) |
| [v4-mmp-transfer](../experiments/results/v4-mmp-transfer) | [доказательства](../experiments/results/v4-mmp-transfer/report.md) |
| [v4-pair-delta](../experiments/results/v4-pair-delta) | [доказательства](../experiments/results/v4-pair-delta/report.md) |
| [v4-pair-delta-submission](../experiments/results/v4-pair-delta-submission) | CSV / конфигурация в каталоге |
| [v4-pair-delta-validation](../experiments/results/v4-pair-delta-validation) | [доказательства](../experiments/results/v4-pair-delta-validation/report.md) |
| [v5-equal-ensemble](../experiments/results/v5-equal-ensemble) | [доказательства](../experiments/results/v5-equal-ensemble/metrics.json) |
| [v5-identity-stress](../experiments/results/v5-identity-stress) | [доказательства](../experiments/results/v5-identity-stress/metrics.json) |
| [v5-pair-anchor-gate](../experiments/results/v5-pair-anchor-gate) | [доказательства](../experiments/results/v5-pair-anchor-gate/report.md) |
| [v5-pair-bagging](../experiments/results/v5-pair-bagging) | [доказательства](../experiments/results/v5-pair-bagging/report.md) |
| [v5-pair-features](../experiments/results/v5-pair-features) | [доказательства](../experiments/results/v5-pair-features/report.md) |
| [v5-pair-regularization](../experiments/results/v5-pair-regularization) | [доказательства](../experiments/results/v5-pair-regularization/report.md) |
| [v5-pair-tailweight](../experiments/results/v5-pair-tailweight) | [доказательства](../experiments/results/v5-pair-tailweight/report.md) |
| [v6-pair-background](../experiments/results/v6-pair-background) | [доказательства](../experiments/results/v6-pair-background/report.md) |
| [v6-pair-external](../experiments/results/v6-pair-external) | [доказательства](../experiments/results/v6-pair-external/report.md) |
| [v6-pair-kernel](../experiments/results/v6-pair-kernel) | [доказательства](../experiments/results/v6-pair-kernel/report.md) |
| [v6-pair-local](../experiments/results/v6-pair-local) | [доказательства](../experiments/results/v6-pair-local/report.md) |
| [v6-pair-metadata](../experiments/results/v6-pair-metadata) | [доказательства](../experiments/results/v6-pair-metadata/metrics.json) |
| [v6-pair-offset](../experiments/results/v6-pair-offset) | [доказательства](../experiments/results/v6-pair-offset/report.md) |
| [v8-nested-validation](../experiments/results/v8-nested-validation) | [доказательства](../experiments/results/v8-nested-validation/metrics.json) |
| [v9-backend-validation](../experiments/results/v9-backend-validation) | [доказательства](../experiments/results/v9-backend-validation/metrics.json) |
| [v9-similar-snapshot](../experiments/results/v9-similar-snapshot) | CSV / конфигурация в каталоге |
| [graph_confirmation](../results/graph_confirmation) | [доказательства](../results/graph_confirmation/metrics.json) |
| [graph_final7](../results/graph_final7) | [доказательства](../results/graph_final7/metrics.json) |
| [graph_gated](../results/graph_gated) | [доказательства](../results/graph_gated/metrics.json) |
| [graph_screen](../results/graph_screen) | [доказательства](../results/graph_screen/metrics.json) |
| [graph_submission](../results/graph_submission) | CSV / конфигурация в каталоге |
