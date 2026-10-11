"""Inference-respecting L-bisimulation via colour refinement.

GMV (Fan et al.) matches a query node ``v_t`` to a counterpart ``v'`` in a memory graph when
the two cannot be told apart by ``L`` rounds of message passing: equal input features, and
neighbourhoods that match recursively for ``L`` hops. When that holds, a deterministic
``L``-layer GNN gives both nodes the same output, so the cached output of ``v'`` answers the
query exactly (the paper's Theorem 2 / Corollary 4).

We compute this as *graded* (counting) bisimulation: colour refinement where a node's
round-``r`` colour hashes its round-``r-1`` colour together with the **multiset** of its
neighbours' round-``r-1`` colours. Multiplicities matter for sum/mean/GCN aggregation. The
paper records them as edge "scaling factors" ``S_e``. Here they are part of the colour, so equal
colours already imply equal aggregated messages.

Rounds needed: a node's ``L``-layer output depends on its ``L``-hop neighbourhood, so ``L``
rounds suffice for mean/sum/attention layers (GraphSAGE, GAT). GCN's symmetric normalisation
also uses the degrees of the nodes ``L`` hops away, which costs one extra round
(``rounds_for(L, "gcn") = L + 1``). The paper instead stores degree scaling factors for GCN
(its Table I). The extra round is the conservative, metadata-free equivalent. Any other layer
(including one you registered yourself) also gets the extra round, because extra rounds
only make matching stricter, never wrong.

Colours are stable 64-bit hashes (blake2b), so views materialised offline can be matched
against queries computed later or in another process.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from typing import Any

import torch

from examples.gmv.graph import adjacency_lists

Colour = int
DIGEST_BYTES = 8
NEIGHBOUR_DEGREE_FREE_CONVS = frozenset(
    {
        "sage",
        "gat",  # toolbox reference layers
        "pyg_sage",
        "pyg_gat",
        "pyg_gatv2",
        "pyg_gin",
        "pyg_transformer",  # PyG layers
    }
)
PYG_MODEL_CONVS = {"GCN": "pyg_gcn", "GraphSAGE": "pyg_sage", "GAT": "pyg_gat", "GIN": "pyg_gin"}


def conv_name(conv: str | Mapping[str, Any]) -> str:
    """Layer name from a spec: ``"gcn"`` or ``{"name": "gat", "heads": 8}``."""
    return conv if isinstance(conv, str) else str(conv["name"])


def model_conv(model_spec: Mapping[str, Any]) -> str | Mapping[str, Any]:
    """The message-passing layer of a model spec (toolbox ``conv`` slot or a PyG model).

    Unknown models return ``"custom"``, which gets the conservative ``L + 1`` rounds.
    """
    if "conv" in model_spec:
        conv: str | Mapping[str, Any] = model_spec["conv"]
        return conv
    if model_spec.get("name") == "pyg_model":
        return PYG_MODEL_CONVS.get(str(model_spec.get("model", "GCN")), "custom")
    return "custom"


def rounds_for(num_layers: int, conv: str | Mapping[str, Any] = "gcn") -> int:
    """Refinement rounds that make equal colours imply equal outputs for this layer type."""
    extra = 0 if conv_name(conv) in NEIGHBOUR_DEGREE_FREE_CONVS else 1
    return num_layers + extra


def _digest(payload: bytes) -> Colour:
    return int.from_bytes(hashlib.blake2b(payload, digest_size=DIGEST_BYTES).digest(), "big")


def feature_colours(x: torch.Tensor) -> list[Colour]:
    """Round 0: nodes with byte-identical feature vectors share a colour."""
    rows = x.detach().to(torch.float32).cpu().contiguous().numpy()
    return [_digest(row.tobytes()) for row in rows]


def refine(x: torch.Tensor, edge_index: torch.Tensor, rounds: int) -> list[list[Colour]]:
    """Colours for rounds ``0..rounds``; ``result[r][v]`` is node ``v``'s round-``r`` colour."""
    neighbours = adjacency_lists(edge_index, x.shape[0])
    history = [feature_colours(x)]
    for _ in range(rounds):
        prev = history[-1]
        current = []
        for v, nbs in enumerate(neighbours):
            multiset = sorted(prev[u] for u in nbs)
            current.append(_digest(struct.pack(f">Q{len(multiset)}Q", prev[v], *multiset)))
        history.append(current)
    return history


def deepest_valid_round(distance: int, rounds: int) -> int:
    """In a subgraph cut at ``rounds`` hops around a seed, a node at ``distance`` has a correct
    colour only up to round ``rounds - distance`` (beyond that its neighbourhood is truncated)."""
    return max(rounds - distance, 0)
