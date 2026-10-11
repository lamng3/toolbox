"""Message-passing graph layers in pure PyTorch, registered in ``GRAPH_CONVS``.

Graphs use the PyG convention: ``edge_index`` is a ``(2, E)`` long tensor of ``(source, target)``
pairs, node features ``x`` are ``(N, F)``. Aggregation uses ``index_add_`` so no
``torch_scatter`` / ``torch_geometric`` install is needed; :mod:`toolbox.data.adapters` converts
PyG ``Data`` objects to this format when PyG *is* available.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from toolbox.core.constants import EPS
from toolbox.core.registry import GRAPH_CONVS


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    loops = torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
    return torch.cat([edge_index, loops], dim=1)


def degree(index: torch.Tensor, num_nodes: int, dtype: torch.dtype = torch.float) -> torch.Tensor:
    out = torch.zeros(num_nodes, dtype=dtype, device=index.device)
    return out.index_add_(0, index, torch.ones_like(index, dtype=dtype))


def scatter_sum(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros(num_nodes, src.shape[-1], dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax of ``scores`` (E, H) grouped by target node ``index`` (E,)."""
    heads = scores.shape[-1]
    max_per_node = torch.full((num_nodes, heads), -torch.inf, device=scores.device)
    max_per_node = max_per_node.scatter_reduce(
        0, index[:, None].expand_as(scores), scores, reduce="amax", include_self=True
    )
    exp = (scores - max_per_node[index]).exp()
    denom = torch.zeros(num_nodes, heads, device=scores.device).index_add_(0, index, exp)
    return exp / (denom[index] + EPS)


@GRAPH_CONVS.register("gcn")
class GCNConv(nn.Module):
    """Graph convolution (Kipf & Welling, 2017) with symmetric normalisation."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        edge_index = add_self_loops(edge_index, num_nodes)
        src, dst = edge_index
        deg = degree(dst, num_nodes, x.dtype)
        norm = deg.pow(-0.5)[src] * deg.pow(-0.5)[dst]
        h = self.linear(x)
        return scatter_sum(h[src] * norm[:, None], dst, num_nodes)


@GRAPH_CONVS.register("sage")
class SAGEConv(nn.Module):
    """GraphSAGE with mean aggregation (Hamilton et al., 2017)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        src, dst = edge_index
        summed = scatter_sum(x[src], dst, num_nodes)
        mean = summed / degree(dst, num_nodes, x.dtype).clamp(min=1)[:, None]
        return self.self_linear(x) + self.neigh_linear(mean)


@GRAPH_CONVS.register("gat")
class GATConv(nn.Module):
    """Graph attention (Veličković et al., 2018), multi-head with concatenated outputs."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, negative_slope: float = 0.2):
        super().__init__()
        if out_dim % heads:
            raise ValueError(f"out_dim={out_dim} must be divisible by heads={heads}")
        self.heads, self.head_dim = heads, out_dim // heads
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        edge_index = add_self_loops(edge_index, num_nodes)
        src, dst = edge_index
        h = self.linear(x).view(num_nodes, self.heads, self.head_dim)
        scores = (h * self.att_src).sum(-1)[src] + (h * self.att_dst).sum(-1)[dst]
        alpha = scatter_softmax(F.leaky_relu(scores, self.negative_slope), dst, num_nodes)
        messages = (h[src] * alpha[..., None]).flatten(1)
        return scatter_sum(messages, dst, num_nodes)
