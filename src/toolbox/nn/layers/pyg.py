"""PyTorch Geometric layers as toolbox graph convs (pattern: *Adapter*).

PyG is the standard GNN library, so there is no need to reimplement its layers. Any
``torch_geometric.nn`` message-passing layer can fill a ``conv`` slot:

.. code-block:: yaml

    model:
      name: gnn_node_classifier
      conv: {name: pyg, layer: GATv2Conv, heads: 4}   # any torch_geometric.nn layer
      # or the shortcuts: pyg_gcn, pyg_sage, pyg_gat, pyg_gatv2, pyg_gin, pyg_transformer

The adapter keeps toolbox's ``(in_dim, out_dim)`` contract. For multi-head layers that
concatenate heads, ``out_dim`` is the total width, so PyG gets ``out_dim // heads`` per head.
When ``out_dim`` isn't divisible by ``heads`` (typically the output layer of a classifier), heads
are averaged instead (``concat=False``), as in the GAT paper's output layer.
Install with ``pip install 'toolbox[gnn]'``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from toolbox.core.decorators import require
from toolbox.core.registry import GRAPH_CONVS

PYG_NN = "torch_geometric.nn"
GNN_EXTRA = "gnn"
MULTI_HEAD_KEY = "heads"
GIN_HIDDEN_MULT = 2


def _gin_mlp(in_dim: int, out_dim: int) -> nn.Module:
    hidden = GIN_HIDDEN_MULT * out_dim
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


@GRAPH_CONVS.register("pyg")
class PyGConv(nn.Module):
    """Any ``torch_geometric.nn`` layer by class name, e.g. ``layer="GATv2Conv"``.

    Extra keyword arguments go to the PyG constructor unchanged.
    """

    def __init__(self, in_dim: int, out_dim: int, layer: str = "GCNConv", **kwargs: Any) -> None:
        super().__init__()
        pyg_nn = require(PYG_NN, extra=GNN_EXTRA)
        cls = getattr(pyg_nn, layer)
        if layer == "GINConv":  # GIN wraps an MLP instead of taking channel sizes
            self.layer = cls(_gin_mlp(in_dim, out_dim), **kwargs)
            return
        heads = kwargs.get(MULTI_HEAD_KEY, 1)
        if heads > 1 and kwargs.get("concat", True) and out_dim % heads:
            kwargs["concat"] = False  # average heads, as in GAT's output layer
        concat = heads > 1 and kwargs.get("concat", True)
        self.layer = cls(in_dim, out_dim // heads if concat else out_dim, **kwargs)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layer(x, edge_index)
        return out


def _shortcut(name: str, layer: str, doc: str) -> None:
    def factory(in_dim: int, out_dim: int, **kwargs: Any) -> PyGConv:
        return PyGConv(in_dim, out_dim, layer=layer, **kwargs)

    factory.__doc__ = doc
    factory.__name__ = name
    GRAPH_CONVS.add(name, factory)


_shortcut("pyg_gcn", "GCNConv", "PyG ``GCNConv`` (Kipf & Welling, 2017).")
_shortcut("pyg_sage", "SAGEConv", "PyG ``SAGEConv`` (Hamilton et al., 2017).")
_shortcut(
    "pyg_gat", "GATConv", "PyG ``GATConv`` (Veličković et al., 2018); ``heads`` concatenated."
)
_shortcut("pyg_gatv2", "GATv2Conv", "PyG ``GATv2Conv`` (Brody et al., 2022).")
_shortcut("pyg_gin", "GINConv", "PyG ``GINConv`` with a 2-layer MLP (Xu et al., 2019).")
_shortcut("pyg_transformer", "TransformerConv", "PyG ``TransformerConv`` (Shi et al., 2021).")
