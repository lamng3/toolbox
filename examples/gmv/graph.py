"""Small graph utilities used by the GMV example (pure PyTorch, PyG ``edge_index`` layout)."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import torch


def adjacency_lists(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Neighbour lists of an undirected graph given as directed edge pairs."""
    neighbours: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        neighbours[dst].append(src)
    return neighbours


def hop_distances(
    neighbours: list[list[int]], sources: Iterable[int], max_hops: int
) -> dict[int, int]:
    """BFS distances (≤ ``max_hops``) from the nearest source node."""
    dist = {s: 0 for s in sources}
    queue = deque(dist)
    while queue:
        node = queue.popleft()
        if dist[node] == max_hops:
            continue
        for nb in neighbours[node]:
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


@dataclass
class Subgraph:
    """An induced subgraph with local ↔ global id maps."""

    nodes: torch.Tensor  # (n,) global ids; local id i ↔ nodes[i]
    x: torch.Tensor
    edge_index: torch.Tensor  # local ids
    distance: torch.Tensor  # (n,) hop distance from the seed set

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.numel())

    def local_id(self, global_id: int) -> int:
        return int((self.nodes == global_id).nonzero()[0])


def k_hop_subgraph(
    seeds: Iterable[int],
    hops: int,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    neighbours: list[list[int]] | None = None,
) -> Subgraph:
    """The subgraph induced by all nodes within ``hops`` of ``seeds`` (``G^L(V_s)`` in GMV)."""
    neighbours = neighbours or adjacency_lists(edge_index, x.shape[0])
    dist = hop_distances(neighbours, seeds, hops)
    nodes = torch.tensor(sorted(dist), dtype=torch.long)
    local = {int(g): i for i, g in enumerate(nodes.tolist())}
    pairs = [(local[src], local[dst]) for dst in local for src in neighbours[dst] if src in local]
    sub_edges = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs
        else torch.empty(2, 0, dtype=torch.long)
    )
    distance = torch.tensor([dist[int(g)] for g in nodes.tolist()], dtype=torch.long)
    return Subgraph(nodes, x[nodes], sub_edges, distance)
