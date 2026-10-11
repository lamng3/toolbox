"""Numerical parity with PyTorch Geometric, the de-facto reference for GNN layers.

Same weights + same inputs must give the same outputs. Skipped when PyG isn't installed
(``pip install 'toolbox[gnn]'``); CI installs it.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

from toolbox.nn.layers import GATConv, GCNConv, SAGEConv
from toolbox.nn.models import GNNNodeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
pyg = pytest.importorskip("torch_geometric.nn")

ATOL = 1e-5


@pytest.fixture
def graph() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    num_nodes = 30
    edges = torch.randint(num_nodes, (2, 90))
    edges = torch.cat([edges, edges.flip(0)], dim=1).unique(dim=1)  # undirected, with self-loops
    return torch.randn(num_nodes, 8), edges


def randomize(*params: torch.Tensor) -> None:
    with torch.no_grad():
        for p in params:
            p.normal_()


def test_gcn_matches_pyg(graph: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, edge_index = graph
    ours, ref = GCNConv(8, 6), pyg.GCNConv(8, 6)
    randomize(ours.linear.weight, ours.bias)
    with torch.no_grad():
        ref.lin.weight.copy_(ours.linear.weight)
        ref.bias.copy_(ours.bias)
    torch.testing.assert_close(ours(x, edge_index), ref(x, edge_index), atol=ATOL, rtol=0)


def test_sage_matches_pyg(graph: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, edge_index = graph
    ours, ref = SAGEConv(8, 6), pyg.SAGEConv(8, 6)
    with torch.no_grad():
        ref.lin_l.weight.copy_(ours.neigh_linear.weight)
        ref.lin_l.bias.copy_(ours.self_linear.bias)
        ref.lin_r.weight.copy_(ours.self_linear.weight)
    torch.testing.assert_close(ours(x, edge_index), ref(x, edge_index), atol=ATOL, rtol=0)


def test_gat_matches_pyg(graph: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, edge_index = graph
    heads = 2
    ours, ref = GATConv(8, 6, heads=heads), pyg.GATConv(8, 6 // heads, heads=heads)
    randomize(ours.bias)
    with torch.no_grad():
        ref.lin.weight.copy_(ours.linear.weight)
        ref.att_src.copy_(ours.att_src[None])
        ref.att_dst.copy_(ours.att_dst[None])
        ref.bias.copy_(ours.bias)
    torch.testing.assert_close(ours(x, edge_index), ref(x, edge_index), atol=ATOL, rtol=0)


def test_gat_averaged_heads_match_pyg(graph: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Output width not divisible by heads → heads are averaged (PyG ``concat=False``)."""
    x, edge_index = graph
    ours, ref = GATConv(8, 3, heads=2), pyg.GATConv(8, 3, heads=2, concat=False)
    randomize(ours.bias)
    with torch.no_grad():
        ref.lin.weight.copy_(ours.linear.weight)
        ref.att_src.copy_(ours.att_src[None])
        ref.att_dst.copy_(ours.att_dst[None])
        ref.bias.copy_(ours.bias)
    torch.testing.assert_close(ours(x, edge_index), ref(x, edge_index), atol=ATOL, rtol=0)


@pytest.mark.parametrize(
    "conv",
    [
        "pyg_gcn",
        "pyg_sage",
        {"name": "pyg_gat", "heads": 2},
        {"name": "pyg", "layer": "GATv2Conv", "heads": 4},
        "pyg_gin",
        "pyg_transformer",
    ],
)
def test_pyg_layers_fill_toolbox_slots(graph: tuple[torch.Tensor, torch.Tensor], conv) -> None:  # type: ignore[no-untyped-def]
    x, edge_index = graph
    for head in ("linear", None):
        model = GNNNodeClassifier(8, 3, hidden_dim=8, conv=conv, head=head).eval()
        assert model(x, edge_index).shape == (30, 3)


def test_pyg_models_and_any_torch_module_train_through_the_builder(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from toolbox.experiment import ExperimentBuilder

    for spec in (
        {"name": "pyg_model", "model": "GraphSAGE", "hidden_channels": 16, "num_layers": 2},
        {
            "name": "torch_module",
            "target": "torch_geometric.nn.models:GCN",
            "in_channels": 32,
            "hidden_channels": 16,
            "num_layers": 2,
            "out_channels": 4,
        },
    ):
        trainer = (
            ExperimentBuilder("ext", output_dir=str(tmp_path / spec["name"]))
            .with_data("synthetic_graph", num_nodes=60)
            .with_model(spec.pop("name"), **spec)
            .with_objective("node_classification")
            .with_trainer("supervised", max_steps=2, batch_size=1, device="cpu", eval_every=0)
            .with_callbacks()
            .build()
        )
        assert "eval/accuracy" in trainer.fit()


def test_classic_two_layer_gcn_matches_the_reference_model(
    graph: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """``head=None`` is the standard Kipf GCN: conv → relu → conv, logits from the last conv."""
    x, edge_index = graph
    ours = GNNNodeClassifier(
        8, 4, hidden_dim=16, num_layers=2, conv="gcn", dropout=0.0, head=None
    ).eval()
    conv1, conv2 = pyg.GCNConv(8, 16), pyg.GCNConv(16, 4)
    with torch.no_grad():
        for mine, ref in zip(ours.encoder.convs, (conv1, conv2)):
            ref.lin.weight.copy_(mine.linear.weight)
            ref.bias.copy_(mine.bias)
        expected = conv2(F.relu(conv1(x, edge_index)), edge_index)
    torch.testing.assert_close(ours(x, edge_index), expected, atol=ATOL, rtol=0)


def test_glorot_initialisation_like_the_reference() -> None:
    weight = GCNConv(256, 256).linear.weight.detach()
    glorot_std = (2 / (256 + 256)) ** 0.5
    assert abs(float(weight.std()) - glorot_std) < 0.1 * glorot_std
