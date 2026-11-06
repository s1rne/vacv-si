# Замороженный CheMeleon: kernel/SVR и TabICL

Исходный коммит: `2958cdb5b4780845d6bf4d7f880914c95c6a8f61`; ветка
`codex/astra-pretrained-embeddings`. Исследовательский код не меняет production-профили.

## Гипотеза и данные

Замороженный молекулярный encoder может добавить информацию к Morgan и контексту
публикации. Мотивация — [абляции OpenADMET](https://openadmet.ghost.io/are-tabular-foundation-models-all-you-need/):
CheMeleon с табличной моделью оказался полезен на другом endpoint, PXR. Это не
доказательство переноса на VACV и не воспроизведение их ансамбля со скринингом.

Использованы 320 train и 597 test через `vacv.dataset.load()`: запрещённые конкурсные
`CC50_uM`, `IC50_uM_VACV` и их пропуски не доступны моделям. Разрешённый снимок
внешних признаков и литературная подстановка совпадают с baseline. Sentinel —
точное `SI == 10000`; реальный SI=36066.667 остаётся активным и входит в raw MSE.
Хеши входов сохранены в `input-manifest.json`. Внешние данные повторно не собирались.

899 уникальных SMILES train+test преобразованы локально в 2048-мерные векторы.
Encoder заморожен; labels, context и target при извлечении не используются.
Все молекулы распознаны, все векторы конечны. Test не участвует в PCA или обучении
голов; объединение SMILES только устраняет дубли при независимом извлечении.
Гарантировать отсутствие этих соединений в исходном корпусе предобучения нельзя.

## Зафиксированные модели и протокол

До просмотра метрик выбраны четыре головы, без перебора гиперпараметров:

| Голова | Настройки |
|---|---|
| `che-ridge` | PCA64, RBF + 2·same_group + same_group·same_matched, ridge alpha=0.3 |
| `che-additive` | предыдущее ядро плюс Morgan Tanimoto² |
| `che-svr` | ядро `che-ridge`, SVR C=1, epsilon=0.05 |
| `che-tabicl` | PCA32 + strain/Culture/group + пять Title + matched + девять presence; TabICLClassifier |

PCA без whitening и без масштабирования исходных embedding-координат, full SVD;
fit только на train соответствующего фолда. RBF gamma = 1/(число PC · var(train_PC)).
Ridge центрирует бинарные метки по train prior. Вероятности ridge/SVR ограничены
[0,1]. Все головы используют среднее активных только train: p·10000+(1−p)·mean_active.
После этого применяется общий `literature.apply`. Группа — точная пара strain/Culture.
TabICL: 2 estimators, batch_size=1, CPU, use_amp=False, n_jobs=4, random_state=0,
allow_auto_download=False; категориальная обработка — штатная ordinal внутри fit.
Библиотека предупреждает о cardinality >40 у strain/group; кодировка не менялась.

Каждая голова также проверена с фиксированным весом 0.25 к 0.75 `hybrid`.
Screen: random KFold5 seeds 0/1/2 и исходные scaffold5, один model seed hybrid.
PCA, prior и bandwidth не видят validation. Веса смеси не подбираются по OOF.
Сравнение использует те же фолды и общий inference baseline; 36 результатов,
включая отрицательные, находятся в `screen-metrics.json` вместе с SHA256 OOF.

| Профиль | random mean, MSE M | scaffold, MSE M |
|---|---:|---:|
| hybrid | 9.569396 | 11.977887 |
| che-ridge | 10.446973 | 13.608954 |
| che-additive | 10.079156 | 12.812118 |
| che-svr | 10.901628 | 15.015576 |
| che-tabicl | 9.627440 | 12.575304 |
| hybrid + 25% che-ridge | 9.603844 | 12.142391 |
| hybrid + 25% che-additive | 9.597031 | 12.063901 |
| hybrid + 25% che-svr | 9.576802 | 12.301534 |
| hybrid + 25% che-tabicl | 9.483746 | 11.942902 |

CheMeleon+TabICL даёт слабый screening-сигнал только в смеси: −0.895% random mean,
−0.292% scaffold. Остальные головы отвергнуты. Screening не является независимой
оценкой выбранного кандидата. Ни одна голова отдельно не превосходит hybrid по
среднему random и scaffold одновременно. Нового leaderboard score нет.

## Проверка замороженного кандидата

До новых splits зафиксирован `frozen-candidate.json`: 25% того же CheMeleon-PCA32
TabICL (2 estimators, random_state=0) + 75% hybrid, теперь с семью model seeds
контекстного леса. Никакие веса или параметры после просмотра confirmation не
менялись. Отдельно добавлено заранее выбранное в параллельном исследовании мягкое
литературное правило: для найденных молекул p·10000+(1−p)·literature_SI, а для
ненайденных остаётся raw prediction. Используется итоговая смешанная вероятность;
готовые hard-подстановки не являются входом мягкого правила.

| Разбиение | hybrid hard | смесь hard | hybrid soft | смесь soft |
|---|---:|---:|---:|---:|
| random-3 | 8.803503 | 8.651893 | 8.656653 | 8.506737 |
| random-4 | 9.049160 | 8.948461 | 8.897749 | 8.802075 |
| random-5 | 8.977600 | 8.896279 | 8.922895 | 8.814812 |
| random mean | 8.943421 | 8.832211 | 8.825766 | 8.707875 |
| scaffold | 11.965575 | 11.934881 | 11.969000 | 11.882621 |

MSE в миллионах. Смесь улучшила все три random splits при обоих правилах:
−1.24% относительно hard hybrid и −1.34% относительно soft hybrid. Scaffold
улучшен на 0.26% hard и 0.72% soft. Полные 24 метрики, включая чистую голову,
находятся в `confirmation-metrics.json`; чистая голова снова хуже по среднему.

Рекомендуемый исследовательский кандидат — 25% CheTabICL + 75% hybrid7 с soft.
Это устойчивый, но небольшой выигрыш на новых разбиениях тех же 320 строк.
Scaffold переиспользован; новых независимых данных нет. Мягкое правило выбрано
в параллельном анализе, поэтому его прибавка не является новой слепой проверкой.
Голова использует всего 2 estimator TabICL; расширение ансамбля здесь не проверялось.

## Веса, окружение и воспроизведение

CheMeleon checkpoint: [Zenodo](https://zenodo.org/records/15460715), MIT,
34,859,448 bytes; MD5 `6a80b54fdb7de37ef0374d302f01e8ce`, SHA256 и embedding SHA
в `embedding-manifest.json`. Реализация соответствует [авторскому API](https://raw.githubusercontent.com/JacksonBurns/chemeleon/b5e5bd88a7070c3f278e7e65e85b792617bf125c/chemeleon_fingerprint.py).
TabICL classifier v2: [Hugging Face](https://huggingface.co/jingang/TabICL/tree/4dcd344ece2c00be9e831fdd35bed57b5ad83e19),
BSD-3-Clause, 110,368,038 bytes; SHA256
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.
Оба checkpoint проверяются кодом до загрузки. Веса и embeddings остаются ignored.

Chemprop 2.3.1 через cuik-molmaker-pin устанавливает RDKit2026.3.6, поэтому encoder
и головы запускались раздельно. Encoder: локальная `.venv`, RDKit2026.3.6.
Головы: Python из соседнего исследовательского worktree `vacv-si-novel/.venv`,
использован только для чтения, RDKit2026.9.1, как у исходного baseline.
Python3.13.14, Torch2.15.0, TabICL2.2.0. Полные снимки pip freeze:
`encoder-requirements.txt`, `heads-requirements.txt`. Воспроизведение в двух
изолированных каталогах, находящихся вне Git:

```bash
uv venv --python 3.13 /tmp/vacv-che-encoder
uv pip install --python /tmp/vacv-che-encoder/bin/python -r experiments/results/2026-11-07-chemeleon/encoder-requirements.txt
uv venv --python 3.13 /tmp/vacv-che-heads
uv pip install --python /tmp/vacv-che-heads/bin/python -r experiments/results/2026-11-07-chemeleon/heads-requirements.txt
mkdir -p data/raw/models
curl --fail --location https://zenodo.org/api/records/15460715/files/chemeleon_mp.pt/content -o data/raw/models/chemeleon_mp.pt
curl --fail --location https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt -o data/raw/models/tabicl-classifier-v2-20260212.ckpt
export PYTHONPATH=.
DYLD_LIBRARY_PATH=/tmp/vacv-che-encoder/lib/python3.13/site-packages/sklearn/.dylibs /tmp/vacv-che-encoder/bin/python experiments/chemeleon_features.py
export DYLD_LIBRARY_PATH=/tmp/vacv-che-heads/lib/python3.13/site-packages/sklearn/.dylibs
/tmp/vacv-che-heads/bin/python experiments/pretrained_embeddings.py
/tmp/vacv-che-heads/bin/python experiments/pretrained_embeddings.py --profiles che-tabicl --seeds 7 --soft --splits random-3 random-4 random-5 scaffold --out outputs/chemeleon-confirmation
/tmp/vacv-che-heads/bin/python experiments/submit_pretrained_embeddings.py --out outputs/submission-chemeleon-soft.csv
/tmp/vacv-che-heads/bin/python -m pytest -q
git diff --check
```

Проверено: 12 новых тестов изоляции PCA, точного sentinel, kernel, finite output,
allowlist TabICL, экспорта без train index и soft по итоговой вероятности;
общий suite 120 passed. На реальном checkpoint отдельно проверены
invalid/empty SMILES (нулевой вектор с сохранением строки) и инвариантность к размеру
batch1/2 (`rtol=1e-5, atol=1e-6`). Первый screen остановился на экспорте из-за отсутствия
`index` в train; после исправления на позиционный ID полностью повторён успешно.

Сохранён `submission-chemeleon-soft.csv`: 597 строк, порядок и множество ID
проверены по sample_submit, значения конечные и неотрицательные. SHA256
`f1db71aac37dea672060990712759fc4bfff047ca4c5a72a80b21cd95d20c2c4`.
В `submission-chemeleon-soft-components.csv` находятся исходные вероятности
hybrid/head, итоговая вероятность, raw и soft прогнозы; метаданные — соседний JSON.
Kaggle API, push и новая загрузка внешних assay-данных не выполнялись.
