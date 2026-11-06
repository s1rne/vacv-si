"""Файнтюнинг химической языковой модели.

Одна и та же обёртка решает три задачи: классификацию неактивных на выборке соревнования,
многозадачную регрессию log IC50 и log CC50 на внешнем наборе (предобучение) и
классификацию поверх уже дообученного тела (перенос).

Голова — среднее по токенам с учётом маски и линейный слой. Пулинг взят тот же, что
у замороженных эмбеддингов, чтобы файнтюнинг сравнивался с линейным зондом при прочих
равных. При предсказании можно усреднять несколько случайных записей одной молекулы.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .augment import expand, randomized
from .embed import MAX_LENGTH, MODELS, REMOTE_CODE, REVISION


@dataclass
class Config:
    model: str = "chemberta-77m-mtr"
    epochs: int = 20
    learning_rate: float = 3e-5
    head_learning_rate: float | None = None
    batch_size: int = 16
    weight_decay: float = 0.01
    dropout: float = 0.1
    warmup: float = 0.1
    augment: int = 1
    tta: int = 1
    seed: int = 0
    freeze_backbone: bool = False
    label: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        parts = [self.model, f"ep{self.epochs}", f"lr{self.learning_rate:g}"]
        if self.augment > 1:
            parts.append(f"aug{self.augment}")
        if self.tta > 1:
            parts.append(f"tta{self.tta}")
        return " ".join(parts)


def _torch():
    import torch

    return torch


def build(config: Config, outputs: int, state: dict | None = None):
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    repo = MODELS[config.model]
    trust = config.model in REMOTE_CODE
    revision = REVISION.get(config.model)
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust, revision=revision)
    backbone = AutoModel.from_pretrained(repo, trust_remote_code=trust, revision=revision)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(nn.Dropout(config.dropout),
                                      nn.Linear(backbone.config.hidden_size, outputs))

        def forward(self, ids, mask):
            hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
            return self.head(pooled)

    model = Model()
    if state is not None:
        model.backbone.load_state_dict(state)
    return tokenizer, model


def _batches(tokenizer, texts: list[str], targets: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    torch = _torch()
    order = np.arange(len(texts))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        encoded = tokenizer([texts[i] for i in index], padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
        yield encoded, torch.as_tensor(targets[index], dtype=torch.float32)


def train(config: Config, texts: list[str], targets: np.ndarray, task: str,
          state: dict | None = None):
    """`task` — "binary" (одна логит-голова) или "regression" (несколько выходов, NaN пропускаются)."""
    import torch
    import torch.nn as nn
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(config.seed)
    targets = np.asarray(targets, dtype=np.float64)
    outputs = 1 if task == "binary" else targets.shape[1]
    tokenizer, model = build(config, outputs, state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()

    if config.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

    head_lr = config.head_learning_rate or config.learning_rate * 10
    groups = [{"params": model.head.parameters(), "lr": head_lr}]
    if not config.freeze_backbone:
        groups.append({"params": model.backbone.parameters(), "lr": config.learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)

    if config.augment > 1:
        texts, targets = expand(texts, targets, config.augment, seed=config.seed)
    steps = max(1, config.epochs * int(np.ceil(len(texts) / config.batch_size)))
    schedule = get_linear_schedule_with_warmup(optimizer, int(steps * config.warmup), steps)

    for epoch in range(config.epochs):
        for encoded, batch in _batches(tokenizer, texts, targets, config.batch_size, True,
                                       config.seed * 1000 + epoch):
            encoded = {key: value.to(device) for key, value in encoded.items()}
            batch = batch.to(device)
            logits = model(encoded["input_ids"], encoded["attention_mask"])
            if task == "binary":
                loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), batch)
            else:
                known = torch.isfinite(batch)
                loss = nn.functional.mse_loss(logits[known], batch[known])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
    return tokenizer, model


def predict(config: Config, tokenizer, model, texts: list[str]) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    model.eval()
    variants = [randomized(text, config.tta, seed=config.seed + index) if config.tta > 1 else [text]
                for index, text in enumerate(texts)]
    flat = [variant for group in variants for variant in group]
    collected = []
    with torch.no_grad():
        for start in range(0, len(flat), 64):
            encoded = tokenizer(flat[start:start + 64], padding=True, truncation=True,
                                max_length=MAX_LENGTH, return_tensors="pt").to(device)
            collected.append(model(encoded["input_ids"], encoded["attention_mask"]).float().cpu().numpy())
    values = np.vstack(collected)
    grouped = values.reshape(len(texts), config.tta, -1).mean(axis=1)
    return grouped


def logits_to_probability(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values[:, 0]))
