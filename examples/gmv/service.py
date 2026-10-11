"""GVInf: answer node-level inference queries from views (the GMV online workflow).

For a query ``Q(M, v_t, G)``:

1. **Match.** Extract the bounded computation neighbourhood ``G^R(v_t)`` (``R`` refinement
   rounds: ``L``, or ``L + 1`` for GCN) and colour it.
   This is the "once-for-all target-local access" the paper allows the graph owner.
2. **Select.** A :class:`~examples.gmv.selection.ViewSelector` picks at most ``k`` views that
   cover ``N^L(v_t)``.
3. **Answer**, cheapest route first:

   * ``view_exact``: a selected view holds a node with ``v_t``'s full ``R``-round colour.
     Return its cached output. This is exact by Corollary 4, and the model is never invoked.
   * ``view_approx``: the deepest matching round is ``r < R`` (and ``r ≥ min_rounds``).
     Return the mean output of the matching nodes. This is approximate, and the *certificate*
     is ``r``: the query and the counterpart agree on ``r`` rounds of message passing.
   * ``fallback``: coverage is insufficient. Run ``M`` on ``G^R(v_t)`` directly. This is
     exact, because an L-layer GNN only sees L hops (+1 for GCN's degrees).

Each answer is logged as an atomic action through :class:`toolbox.tracing.Tracer`, so the
routes, latencies and certificates can be inspected (``toolbox traces``) or used as training
data later.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from examples.gmv.bisim import deepest_valid_round, refine
from examples.gmv.graph import adjacency_lists, k_hop_subgraph
from examples.gmv.selection import Target, ViewSelector
from examples.gmv.views import GraphModelView
from toolbox.tracing import ActionKind, Tracer

MS_PER_S = 1000.0
VIEW_EXACT, VIEW_APPROX, FALLBACK = "view_exact", "view_approx", "fallback"


@dataclass
class Answer:
    node: int
    logits: torch.Tensor
    route: str
    matched_rounds: int
    views_used: int
    latency_ms: float


class GMVService:
    """Serves ``Q(M, v_t, G)`` for one model ``M`` over one graph ``G`` from a set of views."""

    def __init__(
        self,
        model: nn.Module,
        graph: dict[str, torch.Tensor],
        views: Sequence[GraphModelView],
        selector: ViewSelector,
        num_layers: int,
        rounds: int,
        k: int = 5,
        min_rounds: int | None = None,
        tracer: Tracer | None = None,
        agent: str = "gmv",
    ) -> None:
        self.model = model.eval()
        self.graph = graph
        self.views = list(views)
        self.selector = selector
        self.num_layers = num_layers
        self.rounds = rounds
        self.k = k
        self.min_rounds = self.rounds if min_rounds is None else min_rounds
        self.tracer, self.agent = tracer, agent
        self.neighbours = adjacency_lists(graph["edge_index"], graph["x"].shape[0])
        self.clock = 0
        self.routes: Counter[str] = Counter()

    def _targets(self, distance: torch.Tensor, colours: list[list[int]]) -> list[Target]:
        """Coverage targets: nodes within L hops, at their deepest valid round."""
        targets = []
        for v, d in enumerate(distance.tolist()):
            if d <= self.num_layers:
                r = deepest_valid_round(d, self.rounds)
                targets.append((v, r, colours[r][v]))
        return targets

    @torch.no_grad()
    def answer(self, node: int) -> Answer:
        self.clock += 1
        start = time.perf_counter()
        sub = k_hop_subgraph(
            [node], self.rounds, self.graph["x"], self.graph["edge_index"], self.neighbours
        )
        seed = sub.local_id(node)
        colours = refine(sub.x, sub.edge_index, self.rounds)
        seed_colours = [colours[r][seed] for r in range(self.rounds + 1)]
        selected = self.selector.select(
            self.views, seed_colours, self._targets(sub.distance, colours), self.k, self.clock
        )

        logits, route, matched = self._from_views(selected, colours, seed)
        if logits is None:
            logits = self.model(sub.x, sub.edge_index)[seed]
            route = FALLBACK
        latency = (time.perf_counter() - start) * MS_PER_S
        self.routes[route] += 1
        answer = Answer(node, logits, route, matched, len(selected), latency)
        self._trace(answer)
        return answer

    def _from_views(
        self, selected: Sequence[GraphModelView], colours: list[list[int]], seed: int
    ) -> tuple[torch.Tensor | None, str, int]:
        full = colours[self.rounds][seed]
        for view in selected:
            hit = view.lookup(full)
            if hit is not None:
                self._mark_used(view)
                return hit, VIEW_EXACT, self.rounds
        for r in range(self.rounds - 1, self.min_rounds - 1, -1):
            matches = [
                (v, v.partial[r][colours[r][seed]])
                for v in selected
                if colours[r][seed] in v.partial[r]
            ]
            if matches:
                for view, _ in matches:
                    self._mark_used(view)
                return torch.stack([m for _, m in matches]).mean(0), VIEW_APPROX, r
        return None, FALLBACK, 0

    def _mark_used(self, view: GraphModelView) -> None:
        view.last_used, view.uses = self.clock, view.uses + 1

    def _trace(self, answer: Answer) -> None:
        if self.tracer is None:
            return
        self.tracer.record(
            ActionKind.STEP,
            agent=self.agent,
            state={"node": answer.node},
            action=answer.route,
            latency_ms=answer.latency_ms,
            metadata={
                "matched_rounds": answer.matched_rounds,
                "views_used": answer.views_used,
                "prediction": int(answer.logits.argmax()),
            },
        )
