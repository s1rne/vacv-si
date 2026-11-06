"""Графовые сети с нуля: GIN и AttentiveFP, с дескрипторами RDKit и без них."""

from __future__ import annotations

from vacv.neural import store
from vacv.neural.gnn import GnnConfig, evaluate
from vacv.protocol import HEADER, task

PLAN = [
    GnnConfig(architecture="gin", use_descriptors=False, epochs=80),
    GnnConfig(architecture="gin", use_descriptors=True, epochs=80),
    GnnConfig(architecture="gin", use_descriptors=True, epochs=200, label="GIN + RDKit ep200"),
    GnnConfig(architecture="attentivefp", use_descriptors=False, epochs=80),
    GnnConfig(architecture="attentivefp", use_descriptors=True, epochs=80),
    GnnConfig(architecture="attentivefp", use_descriptors=True, epochs=200,
              label="ATTENTIVEFP + RDKit ep200"),
]


def main() -> None:
    job = task()
    results = []
    print(HEADER)
    for config in PLAN:
        for split in ("random", "scaffold"):
            result = evaluate(job, split, config)
            results.append(result)
            print(result.row(), flush=True)
    store.save("graphs", results)


if __name__ == "__main__":
    main()
