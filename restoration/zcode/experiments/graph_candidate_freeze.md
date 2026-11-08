# Замороженный кандидат перед confirmation

2026-11-07. База `2958cdb`, ветка `codex/astra-graph-correction`.

Кандидат выбран **после** исследования random split seeds 0, 1, 2 и scaffold.
Обычная residual diffusion не улучшила среднее random MSE. Затем проверены две
post-hoc exploratory confidence gates (beta 0.25 и 0.5). Только beta 0.25 улучшила
все три random-разбиения: 9.499572 / 9.802173 / 9.333624M против hybrid
9.514975 / 9.843005 / 9.350207M; среднее −0.254%. Это выбор по screening, не confirmation.

До просмотра random split seeds 3, 4, 5 фиксируется **один** вариант:

- Base: `predict_ensemble(..., seeds=1, profile="hybrid")`, то есть один model seed 0.
- Inner OOF: KFold(4, shuffle=True, random_state=2718), независимо внутри outer train.
- Residual: `(y == 10000) - inner_oof_probability`; настоящий 36066.667 — активный.
- Graph: Morgan Tanimoto³, k=6, cross_group=0.05, weak complete-group edges 0.1.
- Геометрия включает разрешённые признаки 597 test-строк; test labels не используются.
- Diffusion: `(D_uu + 0.03 I - W_uu) delta_u = W_ul residual_l`.
- Correction: `clip(p + 0.25 * 4*p*(1-p) * delta, 0, 1)`.
- SI: активное среднее только outer train; исходная hard literature substitution.
- Confirmation: outer KFold seeds **3, 4, 5**, по пять фолдов, тот же model seed 0.
- Никакого выбора beta, графа, inner splits или варианта по confirmation.

Команда confirmation: `PYTHONPATH=. uv run python experiments/graph_correction.py
--candidate gated_diffusion_0.25 --splits 3 4 5 --seeds 1 --out results/graph_confirmation`.
При исчезновении выигрыша фиксируется отрицательный результат; изменение модели
после confirmation не будет представляться как независимое подтверждение.

## Финальная проверка ансамбля, план до её запуска

Single-seed confirmation завершилась: gated 8.602210 / 8.938484 / 9.011617M,
hybrid 8.815902 / 9.060875 / 9.003592M; среднее −1.220%, выигрыш на двух split из трёх.
Параметры графа/gate остаются прежними. По согласованному следующему шагу
проверяются **7 model seeds (0–6) и в inner crossfit, и в outer модели** на
тех же confirmation 3, 4, 5 и scaffold. Это проверка финального ансамбля,
не новая независимая выборка и не поиск параметров.

Сравниваются четыре фиксированных режима: hybrid hard, hybrid soft,
gated hard, gated soft. Soft-вариант передан основной веткой, функция
`vacv/postprocess.py:conditional_lookup` скопирована без изменений:
для matched `p*10000 + (1-p)*literature_si`, для unmatched исходный raw SI.
Формула gate, beta, graph/inner параметры не изменяются по итогам этой проверки.

Команда: `PYTHONPATH=. uv run python experiments/graph_correction.py
--candidate gated_diffusion_0.25 --splits 3 4 5 scaffold --seeds 7 --with-soft
--out results/graph_final7`.
