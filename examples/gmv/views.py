"""Graph model views (GMVs): memory graph + memoisation table + cached model outputs.

A GMV ``V = (G_V, T_V, Z_V)`` in the paper packages:

* ``G_V`` — a *memory graph*, e.g. a published subgraph on which the model was run;
* ``T_V`` — metadata to match and reconstruct (here: per-round bisimulation colours);
* ``Z_V`` — the model's outputs over ``G_V``.

A view is *extended* once ``T_V`` and ``Z_V`` are materialised (:func:`materialize_view`).
:func:`minimize` is the GVMin step: nodes in the same final-round colour class are
inference-equivalent, so their outputs collapse to a single entry. ``memo_bytes`` measures the
storage that saves.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from examples.gmv.bisim import Colour, refine
from examples.gmv.graph import Subgraph, k_hop_subgraph

BYTES_PER_COLOUR = 8


@dataclass
class GraphModelView:
    """An extended GMV. ``rounds`` = colour-refinement rounds (``bisim.rounds_for``)."""

    view_id: int
    memory_graph: Subgraph
    rounds: int
    colour_sets: list[set[Colour]]  # per round: colours present in G_V (coverage tests)
    memo: dict[Colour, torch.Tensor] | list[tuple[Colour, torch.Tensor]]  # final round → output
    partial: list[dict[Colour, torch.Tensor]]  # per round < rounds: colour → mean output
    created_at: int = 0
    last_used: int = 0
    uses: int = 0
    minimized: bool = False
    meta: dict[str, object] = field(default_factory=dict)

    def lookup(self, colour: Colour) -> torch.Tensor | None:
        """Cached output of a node whose final-round colour equals ``colour``."""
        if isinstance(self.memo, dict):
            return self.memo.get(colour)
        for c, out in self.memo:  # unminimised: linear scan over every stored node
            if c == colour:
                return out
        return None

    @property
    def memo_entries(self) -> int:
        return len(self.memo)

    def memo_bytes(self) -> int:
        entries = self.memo.values() if isinstance(self.memo, dict) else (o for _, o in self.memo)
        return sum(BYTES_PER_COLOUR + o.numel() * o.element_size() for o in entries)


@torch.no_grad()
def materialize_view(
    view_id: int, model: nn.Module, memory_graph: Subgraph, rounds: int, created_at: int = 0
) -> GraphModelView:
    """Extend a virtual view: run ``M`` over ``G_V`` once, record colours and outputs."""
    colours = refine(memory_graph.x, memory_graph.edge_index, rounds)
    outputs = model(memory_graph.x, memory_graph.edge_index).detach()
    final = colours[rounds]
    partial: list[dict[Colour, torch.Tensor]] = []
    for r in range(rounds):
        groups: dict[Colour, list[torch.Tensor]] = {}
        for v, c in enumerate(colours[r]):
            groups.setdefault(c, []).append(outputs[v])
        partial.append({c: torch.stack(outs).mean(0) for c, outs in groups.items()})
    return GraphModelView(
        view_id=view_id,
        memory_graph=memory_graph,
        rounds=rounds,
        colour_sets=[set(c) for c in colours],
        memo=[(final[v], outputs[v]) for v in range(len(final))],
        partial=partial,
        created_at=created_at,
        last_used=created_at,
    )


def minimize(view: GraphModelView) -> GraphModelView:
    """GVMin (compression): collapse inference-equivalent nodes to one memo entry each."""
    if view.minimized or isinstance(view.memo, dict):
        return view
    collapsed: dict[Colour, torch.Tensor] = {}
    for colour, out in view.memo:
        collapsed.setdefault(colour, out)
    view.memo, view.minimized = collapsed, True
    return view


def sample_views(
    model: nn.Module,
    graph: dict[str, torch.Tensor],
    candidate_nodes: Sequence[int],
    rounds: int,
    num_views: int = 200,
    max_targets: int = 5,
    max_hops: int | None = None,
    seed: int = 0,
    neighbours: list[list[int]] | None = None,
) -> list[GraphModelView]:
    """Offline phase: materialise views over subgraphs around random public nodes.

    In deployment, memory graphs come from published or historical data. Sampling
    ``k``-hop subgraphs mirrors the upstream experiments.
    """
    rng = random.Random(seed)
    max_hops = max_hops or rounds
    model.eval()
    views = []
    for i in range(num_views):
        targets = rng.sample(list(candidate_nodes), rng.randint(1, max_targets))
        sub = k_hop_subgraph(
            targets, rng.randint(1, max_hops), graph["x"], graph["edge_index"], neighbours
        )
        views.append(materialize_view(i, model, sub, rounds, created_at=0))
    return views
