"""Эмбеддинги предобученных химических языковых моделей.

Каждая модель читает SMILES и отдаёт один вектор на молекулу. Берутся два варианта
агрегации: среднее по токенам с учётом маски и вектор первого токена. У MoLFormer-XL
дополнительно `pooler_output` — карточка модели предлагает именно его.

Эмбеддинги считаются один раз и кладутся в `.npz`: обучающая выборка соревнования — 320
молекул, перебор голов и фолдов идёт потом по кешу.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE = Path("data/interim/embeddings")

MODELS = {
    "chemberta-77m-mlm": "DeepChem/ChemBERTa-77M-MLM",
    "chemberta-77m-mtr": "DeepChem/ChemBERTa-77M-MTR",
    "chemberta-10m-mtr": "DeepChem/ChemBERTa-10M-MTR",
    "chemberta-zinc": "seyonec/ChemBERTa-zinc-base-v1",
    "molformer-xl": "ibm-research/MoLFormer-XL-both-10pct",
}
# MoLFormer обрезает вход по max_position_embeddings = 202 и требует своего кода.
# Код в main написан под transformers 5; под 4.x работает только ревизия compat-v4.
REMOTE_CODE = {"molformer-xl"}
REVISION = {"molformer-xl": "compat-v4"}
MAX_LENGTH = 202


@dataclass
class Embedding:
    name: str
    mean: np.ndarray
    cls: np.ndarray
    pooler: np.ndarray | None
    seconds: float

    def variant(self, kind: str) -> np.ndarray:
        if kind == "pooler" and self.pooler is None:
            raise ValueError(f"{self.name} не отдаёт pooler_output")
        return {"mean": self.mean, "cls": self.cls, "pooler": self.pooler}[kind]


def _key(name: str, smiles: list[str]) -> Path:
    digest = hashlib.sha1("\n".join(smiles).encode()).hexdigest()[:12]
    return CACHE / f"{name}.{len(smiles)}.{digest}.npz"


def compute(name: str, smiles: list[str], batch_size: int = 32, device: str | None = None) -> Embedding:
    import time

    import torch
    from transformers import AutoModel, AutoTokenizer

    repo = MODELS[name]
    trust = name in REMOTE_CODE
    revision = REVISION.get(name)
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust, revision=revision)
    model = AutoModel.from_pretrained(repo, trust_remote_code=trust, revision=revision)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    means, heads, poolers = [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            batch = smiles[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH,
                                return_tensors="pt").to(device)
            output = model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            means.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy())
            heads.append(hidden[:, 0].float().cpu().numpy())
            pooled = getattr(output, "pooler_output", None)
            if pooled is not None:
                poolers.append(pooled.float().cpu().numpy())
    elapsed = time.perf_counter() - started
    return Embedding(name, np.vstack(means), np.vstack(heads),
                     np.vstack(poolers) if poolers else None, elapsed)


def load(name: str, smiles: list[str], **kwargs) -> Embedding:
    path = _key(name, smiles)
    if path.exists():
        stored = np.load(path, allow_pickle=False)
        pooler = stored["pooler"] if "pooler" in stored.files else None
        return Embedding(name, stored["mean"], stored["cls"], pooler, float(stored["seconds"]))
    embedding = compute(name, smiles, **kwargs)
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {"mean": embedding.mean, "cls": embedding.cls, "seconds": embedding.seconds}
    if embedding.pooler is not None:
        payload["pooler"] = embedding.pooler
    np.savez(path, **payload)
    return embedding
