"""Graph datasets for node-level pretraining and fine-tuning.

A graph is a plain dict ``{"x", "edge_index", "y", "train_mask", "val_mask", "test_mask"}``,
so models depend on tensors, not on a graph library.
"""

from __future__ import annotations

import torch

from toolbox.core.constants import DEFAULT_SEED
from toolbox.core.registry import DATASETS
from toolbox.data.adapters import pyg_to_graph
from toolbox.data.collators import identity_collate
from toolbox.data.datasets.base import DataBundle, ListDataset

Graph = dict[str, torch.Tensor]

TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2


def random_split_masks(num_nodes: int, generator: torch.Generator) -> dict[str, torch.Tensor]:
    perm = torch.randperm(num_nodes, generator=generator)
    n_train, n_val = int(TRAIN_FRACTION * num_nodes), int(VAL_FRACTION * num_nodes)
    masks = {k: torch.zeros(num_nodes, dtype=torch.bool) for k in ("train", "val", "test")}
    masks["train"][perm[:n_train]] = True
    masks["val"][perm[n_train : n_train + n_val]] = True
    masks["test"][perm[n_train + n_val :]] = True
    return {f"{k}_mask": v for k, v in masks.items()}


def stochastic_block_model(
    num_nodes: int,
    num_classes: int,
    feature_dim: int,
    p_in: float,
    p_out: float,
    feature_noise: float,
    seed: int,
) -> Graph:
    """Communities with denser intra-class edges and class-dependent noisy features."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(num_classes, (num_nodes,), generator=g)
    same = y[:, None] == y[None, :]
    probs = torch.where(same, torch.tensor(p_in), torch.tensor(p_out))
    adj = torch.rand(num_nodes, num_nodes, generator=g) < probs
    adj = torch.triu(adj, diagonal=1)
    adj = adj | adj.T
    edge_index = adj.nonzero().T.contiguous()
    centroids = torch.randn(num_classes, feature_dim, generator=g)
    x = centroids[y] + feature_noise * torch.randn(num_nodes, feature_dim, generator=g)
    return {"x": x, "edge_index": edge_index, "y": y, **random_split_masks(num_nodes, g)}


def _graph_bundle(graph: Graph) -> DataBundle:
    info = {"in_dim": int(graph["x"].shape[1]), "num_classes": int(graph["y"].max()) + 1}
    return DataBundle(ListDataset([graph]), ListDataset([graph]), identity_collate, info)


@DATASETS.register("synthetic_graph")
def synthetic_graph(
    num_nodes: int = 500,
    num_classes: int = 4,
    feature_dim: int = 32,
    p_in: float = 0.05,
    p_out: float = 0.005,
    feature_noise: float = 2.0,
    seed: int = DEFAULT_SEED,
) -> DataBundle:
    """Offline stand-in for Cora-style citation graphs (no download needed)."""
    return _graph_bundle(
        stochastic_block_model(
            num_nodes, num_classes, feature_dim, p_in, p_out, feature_noise, seed
        )
    )


@DATASETS.register("planetoid")
def planetoid(dataset: str = "Cora", root: str = "data/raw/planetoid") -> DataBundle:
    """Cora / CiteSeer / PubMed through PyTorch Geometric (``pip install toolbox[gnn]``)."""
    from toolbox.core.decorators import require

    datasets = require("torch_geometric.datasets", extra="gnn")
    return _graph_bundle(pyg_to_graph(datasets.Planetoid(root, dataset)[0]))
