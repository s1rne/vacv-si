"""Графовые сети, обучаемые с нуля: GIN и AttentiveFP.

Молекула — граф: атомы с признаками валентности, заряда, гибридизации и ароматичности,
связи с типом и признаками сопряжения и цикла. Дескрипторы RDKit подаются отдельным
вектором и приклеиваются к результату пулинга перед головой: так графовая часть описывает
структуру, а дескрипторный блок — физхимию, ровно как в рецепте Chemprop с
`--molecule-featurizers rdkit_2d_normalized`.

Предобученного чекпоинта здесь нет: обе сети учатся с нуля на 320 молекулах. Зато они
читают граф, а не строку, и поэтому не видят манеру записи SMILES — на их числах
сравнение с бустингом ничем не испорчено.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger

from ..features import descriptors, parse
from ..protocol import Result, Task, score

RDLogger.DisableLog("rdApp.*")

ATOMS = [5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]
HYBRIDS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
           Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
BONDS = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def _one_hot(value, options) -> list[float]:
    return [float(value == option) for option in options] + [float(value not in options)]


def atom_features(atom) -> list[float]:
    return (_one_hot(atom.GetAtomicNum(), ATOMS)
            + _one_hot(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
            + _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
            + _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
            + _one_hot(atom.GetHybridization(), HYBRIDS)
            + [float(atom.GetIsAromatic()), float(atom.IsInRing()), atom.GetMass() * 0.01])


def bond_features(bond) -> list[float]:
    return (_one_hot(bond.GetBondType(), BONDS)
            + [float(bond.GetIsConjugated()), float(bond.IsInRing())])


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("CC").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def graph(smiles: str, label: float, global_features: np.ndarray):
    import torch
    from torch_geometric.data import Data

    mol = parse(smiles)
    nodes = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    sources, targets, edges = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_features(bond)
        sources += [start, end]
        targets += [end, start]
        edges += [feature, feature]
    if not sources:
        index = torch.zeros((2, 0), dtype=torch.long)
        attributes = torch.zeros((0, BOND_DIM), dtype=torch.float)
    else:
        index = torch.tensor([sources, targets], dtype=torch.long)
        attributes = torch.tensor(edges, dtype=torch.float)
    return Data(x=nodes, edge_index=index, edge_attr=attributes,
                y=torch.tensor([label], dtype=torch.float),
                g=torch.tensor(global_features, dtype=torch.float).unsqueeze(0))


def raw_descriptors(smiles: list[str]) -> np.ndarray:
    return np.vstack([descriptors(parse(text)) for text in smiles])


def normalize(values: np.ndarray, train_index: np.ndarray) -> np.ndarray:
    """Обрезка по квантилям и стандартизация по обучающей части фолда.

    Статистики считаются только на обучении: контрольные молекулы не должны влиять даже
    на масштаб признаков.
    """
    reference = values[train_index]
    low, high = np.quantile(reference, 0.01, axis=0), np.quantile(reference, 0.99, axis=0)
    clipped = np.clip(values, low, high)
    centre, spread = clipped[train_index].mean(0), clipped[train_index].std(0)
    return (clipped - centre) / np.where(spread > 0, spread, 1.0)


@dataclass
class GnnConfig:
    architecture: str = "gin"
    hidden: int = 128
    layers: int = 3
    epochs: int = 80
    learning_rate: float = 1e-3
    batch_size: int = 32
    dropout: float = 0.2
    use_descriptors: bool = True
    seed: int = 0
    label: str = ""

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        suffix = " + RDKit" if self.use_descriptors else ""
        return f"{self.architecture.upper()}{suffix} ep{self.epochs}"


def build(config: GnnConfig, global_dim: int):
    import torch
    import torch.nn as nn
    from torch_geometric.nn import AttentiveFP, GINEConv, global_add_pool

    class Gin(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(ATOM_DIM, config.hidden)
            self.convolutions = nn.ModuleList()
            for _ in range(config.layers):
                body = nn.Sequential(nn.Linear(config.hidden, config.hidden), nn.ReLU(),
                                     nn.Linear(config.hidden, config.hidden))
                self.convolutions.append(GINEConv(body, edge_dim=BOND_DIM))
            self.head = head(config.hidden + global_dim)

        def forward(self, batch):
            state = self.project(batch.x)
            for convolution in self.convolutions:
                state = torch.relu(convolution(state, batch.edge_index, batch.edge_attr))
            pooled = global_add_pool(state, batch.batch)
            return self.head(torch.cat([pooled, batch.g.view(pooled.size(0), -1)], dim=1))

    class Afp(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = AttentiveFP(in_channels=ATOM_DIM, hidden_channels=config.hidden,
                                    out_channels=config.hidden, edge_dim=BOND_DIM,
                                    num_layers=config.layers, num_timesteps=2, dropout=config.dropout)
            self.head = head(config.hidden + global_dim)

        def forward(self, batch):
            pooled = self.body(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            return self.head(torch.cat([pooled, batch.g.view(pooled.size(0), -1)], dim=1))

    def head(dim: int) -> nn.Module:
        return nn.Sequential(nn.Dropout(config.dropout), nn.Linear(dim, config.hidden), nn.ReLU(),
                             nn.Linear(config.hidden, 1))

    torch.manual_seed(config.seed)
    return {"gin": Gin, "attentivefp": Afp}[config.architecture]()


def out_of_fold(job: Task, split: str, config: GnnConfig) -> tuple[np.ndarray, float]:
    import torch
    import torch.nn as nn
    from torch_geometric.loader import DataLoader

    smiles = list(job.frame["SMILES"])
    raw = raw_descriptors(smiles) if config.use_descriptors else np.zeros((len(smiles), 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probability = np.zeros(len(job.y))
    started = time.perf_counter()
    for train_index, validation_index in job.splits(split):
        block = normalize(raw, train_index) if config.use_descriptors else raw
        graphs = [graph(text, float(label), block[index])
                  for index, (text, label) in enumerate(zip(smiles, job.labels))]
        model = build(config, block.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        loader = DataLoader([graphs[i] for i in train_index], batch_size=config.batch_size, shuffle=True)
        model.train()
        for _ in range(config.epochs):
            for batch in loader:
                batch = batch.to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model(batch).squeeze(-1), batch.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        chunk = []
        with torch.no_grad():
            for batch in DataLoader([graphs[i] for i in validation_index], batch_size=64):
                chunk.append(model(batch.to(device)).squeeze(-1).float().cpu().numpy())
        logits = np.clip(np.concatenate(chunk), -60.0, 60.0)
        probability[validation_index] = 1.0 / (1.0 + np.exp(-logits))
    return probability, time.perf_counter() - started


def evaluate(job: Task, split: str, config: GnnConfig) -> Result:
    probability, elapsed = out_of_fold(job, split, config)
    auc, mse = score(job.labels, job.y, probability, job.active_mean)
    return Result(config.name, split, auc, mse, elapsed, probability)
