"""Оценка файла ответа на всей тестовой выборке без отправки на Kaggle.

Публичный лидерборд считается по первым 179 строкам из 597, поэтому он шумит.
Истинная метка теста восстанавливается из колонок, которые организаторы оставили
в файле.

Это диагностика, а не критерий выбора. Выбирать между вариантами модели по этим
числам означает подгонять решение под ответ окольным путём, поэтому все решения
принимаются по кросс-проверке на обучающей выборке, а здешние числа только
показывают, насколько публичный счёт расходится с полным.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vacv.exact_si import exact_si

PUBLIC_ROWS = 179


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()

    test = pd.read_csv("data/raw/kaggle/test.csv")
    truth = exact_si(test).to_numpy(float)
    public = test["index"].to_numpy() < PUBLIC_ROWS

    print(f"{'файл':46} {'весь тест':>12} {'публичные':>12} {'закрытые':>12}")
    for path in args.files:
        prediction = pd.read_csv(path).set_index("index").reindex(test["index"])["SI VACV"].to_numpy(float)
        error = (prediction - truth) ** 2
        print(f"{path.name:46} {error.mean():12,.0f} {error[public].mean():12,.0f} {error[~public].mean():12,.0f}")


if __name__ == "__main__":
    main()
