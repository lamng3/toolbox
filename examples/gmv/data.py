"""A molecule-like benchmark with lots of repeated local structure.

GMV pays off when the queried graph shares structure with previously served "memory graphs"
(the paper's motivating example is molecular fragments). This generator produces many small
molecules over a handful of discrete atom types, so different molecules contain bisimilar
atoms. Molecules are split into *public* ones (training data and view sources) and *private*
ones (queries), so views never contain the query nodes themselves.

Node label: the number of type-0 ("carbon") neighbours, capped at ``NUM_CLASSES - 1``. It depends
on 1-hop structure, so a 2-layer GNN learns it quickly.
"""

from __future__ import annotations

import random

import torch

from toolbox.core.constants import DEFAULT_SEED
from toolbox.core.registry import DATASETS
from toolbox.data.collators import identity_collate
from toolbox.data.datasets.base import DataBundle, ListDataset

ATOM_TYPE_PROBS = (0.45, 0.25, 0.15, 0.1, 0.05)
NUM_ATOM_TYPES = len(ATOM_TYPE_PROBS)
NUM_CLASSES = 4
RING_CLOSURE_PROB = 0.3


def random_molecule(
    rng: random.Random, min_atoms: int, max_atoms: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """A random tree of atoms with an optional ring closure."""
    n = rng.randint(min_atoms, max_atoms)
    types = rng.choices(range(NUM_ATOM_TYPES), weights=ATOM_TYPE_PROBS, k=n)
    bonds = [(rng.randrange(i), i) for i in range(1, n)]
    if n > 3 and rng.random() < RING_CLOSURE_PROB:
        a, b = rng.sample(range(n), 2)
        if (a, b) not in bonds and (b, a) not in bonds:
            bonds.append((a, b))
    return types, bonds


def molecule_graph(
    num_molecules: int = 400,
    min_atoms: int = 5,
    max_atoms: int = 12,
    public_fraction: float = 0.7,
    seed: int = DEFAULT_SEED,
) -> dict[str, torch.Tensor]:
    """Disjoint union of molecules as one graph dict (``x``, ``edge_index``, ``y``, masks,
    ``molecule``)."""
    rng = random.Random(seed)
    types: list[int] = []
    edges: list[tuple[int, int]] = []
    molecule: list[int] = []
    for mol_id in range(num_molecules):
        offset = len(types)
        atom_types, bonds = random_molecule(rng, min_atoms, max_atoms)
        types.extend(atom_types)
        molecule.extend([mol_id] * len(atom_types))
        for a, b in bonds:
            edges += [(offset + a, offset + b), (offset + b, offset + a)]
    num_nodes = len(types)
    x = torch.nn.functional.one_hot(torch.tensor(types), NUM_ATOM_TYPES).float()
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    carbon = (torch.tensor(types) == 0).float()
    carbon_neighbours = torch.zeros(num_nodes).index_add_(0, edge_index[1], carbon[edge_index[0]])
    y = carbon_neighbours.clamp(max=NUM_CLASSES - 1).long()

    mol = torch.tensor(molecule)
    public = mol < int(public_fraction * num_molecules)
    return {
        "x": x,
        "edge_index": edge_index,
        "y": y,
        "molecule": mol,
        "train_mask": public,
        "val_mask": public,
        "test_mask": ~public,
    }


@DATASETS.register("gmv_molecules")
def gmv_molecules(
    num_molecules: int = 400,
    min_atoms: int = 5,
    max_atoms: int = 12,
    public_fraction: float = 0.7,
    seed: int = DEFAULT_SEED,
) -> DataBundle:
    """Toolbox dataset factory: train on public molecules, evaluate on private ones."""
    graph = molecule_graph(num_molecules, min_atoms, max_atoms, public_fraction, seed)
    info = {"in_dim": NUM_ATOM_TYPES, "num_classes": NUM_CLASSES}
    return DataBundle(ListDataset([graph]), ListDataset([graph]), identity_collate, info)
