"""Подстановка литературного значения в условное среднее, а не жёсткая замена.

Правило выбрано по замерам на обучающей выборке (`experiments/substitution.py`,
`experiments/final_substitution.py`):

- границы и неопределённые измерения подставляются целиком: ошибка модели там
  на три порядка больше (0.2M против 2.0M);
- измеренные значения входят как оценка SI активного соединения, но член
  вероятности заглушки сохраняется: соединение может быть помечено в выборке как
  неактивное независимо от того, что измерено в статье;
- класс sentinel, восстановленный по записи «Not active», не подставляется вовсе:
  модель даёт там 1.10M, а подстановка 3.84M, потому что редкое ложное
  срабатывание стоит 10⁸.
"""

from __future__ import annotations

import numpy as np

from .literature import CENSORED_QUALITY, value_and_quality
from .pipeline import SENTINEL


def conditional_lookup(prediction, probability, frame):
    prediction = np.asarray(prediction, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if prediction.shape != (len(frame),) or probability.shape != prediction.shape:
        raise ValueError("Predictions, probabilities and frame must align")
    if (not np.isfinite(prediction).all() or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()):
        raise ValueError("Finite predictions and probabilities in [0, 1] required")

    literature, quality = value_and_quality(frame)
    found = np.isfinite(literature)
    literature = np.clip(literature, 0.0, SENTINEL)
    result = prediction.copy()
    result[found] = probability[found] * SENTINEL + (1 - probability[found]) * literature[found]
    certain = found & np.isin(quality, CENSORED_QUALITY)
    result[certain] = literature[certain]
    return result
