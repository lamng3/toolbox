"""View selection strategies (GVSel and baselines), registered like any toolbox component.

Each selector gets the query node's colour at every round (``seed``) and the query's *coverage
targets*: ``(node, round, colour)`` triples for the nodes in ``N^L(v_t)``, where ``round`` is
the deepest round whose colour is valid for that node. A view covers a target if it contains a
node with the same colour at that round.

* ``gvsel`` — Algorithm 2. Following the paper's selection strategy, a view is first ranked by
  the largest ``l`` for which it holds an ``l``-bisimilar counterpart of ``v_t``, then by its
  marginal coverage of ``N^L(v_t)``. The loop is greedy max-coverage under a budget of ``k``
  views. Coverage is monotone submodular, so greedy is a (1 − 1/e)-approximation.
* ``gvsel_forgetting`` — the same, with each view's gain weighted by an exponential forgetting
  curve over the time since it was last useful (the paper's workload-oriented cache policy).
* ``random`` — ``k`` random views (the upstream ``GVInf-ran`` baseline).
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence

from examples.gmv.bisim import Colour
from examples.gmv.views import GraphModelView
from toolbox.core.registry import Registry

Target = tuple[int, int, Colour]  # (query-local node id, round, colour)

VIEW_SELECTORS: Registry[ViewSelector] = Registry("view_selector")


NO_MATCH = -1


def coverage(view: GraphModelView, targets: Sequence[Target]) -> set[int]:
    return {node for node, r, colour in targets if colour in view.colour_sets[min(r, view.rounds)]}


def seed_depth(view: GraphModelView, seed: Sequence[Colour]) -> int:
    """Largest ``l`` such that the view holds a node ``l``-bisimilar to the query node."""
    for r in range(min(len(seed), view.rounds + 1) - 1, -1, -1):
        if seed[r] in view.colour_sets[r]:
            return r
    return NO_MATCH


class ViewSelector(ABC):
    @abstractmethod
    def select(
        self,
        views: Sequence[GraphModelView],
        seed: Sequence[Colour],
        targets: Sequence[Target],
        k: int,
        now: int,
    ) -> list[GraphModelView]: ...


@VIEW_SELECTORS.register("gvsel")
class GreedyCoverage(ViewSelector):
    def weight(self, view: GraphModelView, now: int) -> float:
        return 1.0

    def select(
        self,
        views: Sequence[GraphModelView],
        seed: Sequence[Colour],
        targets: Sequence[Target],
        k: int,
        now: int,
    ) -> list[GraphModelView]:
        cover = {v.view_id: coverage(v, targets) for v in views}
        cover = {vid: s for vid, s in cover.items() if s}  # only views that match anything
        by_id = {v.view_id: v for v in views}
        depth = {vid: seed_depth(by_id[vid], seed) for vid in cover}
        chosen: list[GraphModelView] = []
        covered: set[int] = set()

        def score(vid: int) -> tuple[int, float]:
            return depth[vid], len(cover[vid] - covered) * self.weight(by_id[vid], now)

        while cover and len(chosen) < k:
            best = max(cover, key=score)
            if not cover[best] - covered and depth[best] == NO_MATCH:
                break  # nothing left adds coverage or a counterpart of v_t
            chosen.append(by_id[best])
            covered |= cover.pop(best)
        return chosen


@VIEW_SELECTORS.register("gvsel_forgetting")
class ForgettingGreedyCoverage(GreedyCoverage):
    """Gain × ``exp(-λ · (now − last_used))``: favours views that keep proving useful."""

    def __init__(self, decay_rate: float = 0.01) -> None:
        self.decay_rate = decay_rate

    def weight(self, view: GraphModelView, now: int) -> float:
        return math.exp(-self.decay_rate * max(now - view.last_used, 0))


@VIEW_SELECTORS.register("random")
class RandomViews(ViewSelector):
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select(
        self,
        views: Sequence[GraphModelView],
        seed: Sequence[Colour],
        targets: Sequence[Target],
        k: int,
        now: int,
    ) -> list[GraphModelView]:
        return self.rng.sample(list(views), min(k, len(views)))
