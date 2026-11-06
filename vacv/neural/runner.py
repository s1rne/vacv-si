"""Перекрёстная проверка файнтюнинга.

Внутри каждого фолда заново создаётся модель из чекпоинта: иначе обучение на одном фолде
подтекает в следующий. Для переноса тело берётся из предобучения, и предобучение тоже
делается заново на каждом фолде — внешний набор пересекается с выборкой соревнования
по 58 соединениям, и валидационные среди них нужно выкинуть, иначе сеть увидит IC50
той самой молекулы, которую её просят предсказать.
"""

from __future__ import annotations

import time

import numpy as np

from ..features import parse
from ..protocol import Result, Task, score
from . import external
from .finetune import Config, logits_to_probability, predict, train


def _smiles(job: Task, canonical: bool) -> list[str]:
    from rdkit import Chem

    texts = list(job.frame["SMILES"])
    if not canonical:
        return texts
    return [Chem.MolToSmiles(parse(text)) for text in texts]


def classify(job: Task, split: str, config: Config, pretrain: bool = False) -> Result:
    import torch

    smiles = _smiles(job, config.canonical)
    keys = external.keys_of(job.frame)
    source = external.table() if pretrain else None
    probability = np.zeros(len(job.y))
    started = time.perf_counter()
    for train_index, validation_index in job.splits(split):
        state = None
        if pretrain:
            held = {keys[i] for i in validation_index if keys[i]}
            texts, targets = external.pretraining(held, source)
            base = Config(**{**config.__dict__, "epochs": config.extra.get("pretrain_epochs", 10),
                             "learning_rate": config.extra.get("pretrain_lr", config.learning_rate),
                             "augment": 1, "tta": 1, "label": "", "canonical": config.canonical})
            _, warm = train(base, texts, targets, "regression")
            state = {key: value.detach().cpu().clone()
                     for key, value in warm.backbone.state_dict().items()}
            del warm
            torch.cuda.empty_cache()
        tokenizer, model = train(config, [smiles[i] for i in train_index],
                                 job.labels[train_index].astype(float), "binary", state)
        logits = predict(config, tokenizer, model, [smiles[i] for i in validation_index])
        probability[validation_index] = logits_to_probability(logits)
        del model, tokenizer
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    auc, mse = score(job.labels, job.y, probability, job.active_mean)
    return Result(config.name, split, auc, mse, elapsed, probability)
