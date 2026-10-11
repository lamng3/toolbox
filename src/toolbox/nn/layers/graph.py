"""Message-passing graph layers in pure PyTorch, registered in ``GRAPH_CONVS``.

Graphs use the PyG convention: ``edge_index`` is a ``(2, E)`` long tensor of ``(source, target)``
pairs, node features ``x`` are ``(N, F)``. Aggregation uses ``index_add_`` so no
``torch_scatter`` / ``torch_geometric`` install is needed; :mod:`toolbox.data.adapters` converts
PyG ``Data`` objects to this format when PyG *is* available.

The layers follow the standard definitions and are checked for numerical parity with PyTorch
Geometric's ``GCNConv``, ``SAGEConv`` and ``GATConv`` in ``tests/unit/test_graph_parity.py``
(same weights, same inputs, same outputs).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from toolbox.core.constants import EPS
from toolbox.core.registry import GRAPH_CONVS


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Give every node exactly one self-loop (existing ones are not duplicated)."""
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
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
    """Graph convolution (Kipf & Welling, 2017): ``D^-1/2 (A + I) D^-1/2 X W + b``.

    The bias is added *after* aggregation, as in the paper and PyG; adding it before would
    scale it by each node's normalised degree. Weights use Glorot initialisation.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        edge_index = add_self_loops(edge_index, num_nodes)
        src, dst = edge_index
        deg = degree(dst, num_nodes, x.dtype)
        norm = deg.pow(-0.5)[src] * deg.pow(-0.5)[dst]
        h = self.linear(x)
        out = scatter_sum(h[src] * norm[:, None], dst, num_nodes)
        return out + self.bias if self.bias is not None else out


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
    """Graph attention (Veličković et al., 2018), multi-head with concatenated outputs.

    ``out_dim`` is the total width (``heads × head_dim``), so GAT is a drop-in replacement for
    the other convs. In PyG terms this is ``GATConv(in_dim, out_dim // heads, heads=heads)``.
    When ``out_dim`` isn't divisible by ``heads`` (e.g. a classifier's output layer), each head
    produces ``out_dim`` features and the heads are averaged, as in the paper's output layer
    (PyG ``concat=False``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        negative_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.concat = out_dim % heads == 0
        self.heads = heads
        self.head_dim = out_dim // heads if self.concat else out_dim
        self.linear = nn.Linear(in_dim, heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        edge_index = add_self_loops(edge_index, num_nodes)
        src, dst = edge_index
        h = self.linear(x).view(num_nodes, self.heads, self.head_dim)
        scores = (h * self.att_src).sum(-1)[src] + (h * self.att_dst).sum(-1)[dst]
        alpha = scatter_softmax(F.leaky_relu(scores, self.negative_slope), dst, num_nodes)
        messages = h[src] * alpha[..., None]  # (E, heads, head_dim)
        out = scatter_sum(messages.flatten(1), dst, num_nodes)
        if not self.concat:
            out = out.view(num_nodes, self.heads, self.head_dim).mean(dim=1)
        return out + self.bias if self.bias is not None else out
