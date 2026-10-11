"""Tests for the GMV example: the exactness condition, selection, minimisation and serving."""

from __future__ import annotations

import pytest
import torch
from examples.gmv.bisim import refine, rounds_for
from examples.gmv.data import molecule_graph
from examples.gmv.graph import k_hop_subgraph
from examples.gmv.selection import VIEW_SELECTORS, coverage
from examples.gmv.service import FALLBACK, VIEW_EXACT, GMVService
from examples.gmv.views import minimize, sample_views

from toolbox.nn.models import GNNNodeClassifier
from toolbox.tracing import InMemoryTraceStore, Tracer

NUM_LAYERS = 2


@pytest.fixture(scope="module")
def graph() -> dict[str, torch.Tensor]:
    return molecule_graph(num_molecules=120, seed=1)


def make_model(conv: str) -> GNNNodeClassifier:
    torch.manual_seed(0)
    return GNNNodeClassifier(
        5, 4, hidden_dim=16, num_layers=NUM_LAYERS, conv=conv, dropout=0.0
    ).eval()


def max_within_class_gap(outputs: torch.Tensor, colours: list[int]) -> float:
    groups: dict[int, list[int]] = {}
    for v, c in enumerate(colours):
        groups.setdefault(c, []).append(v)
    return max(float((outputs[vs] - outputs[vs[0]]).abs().max()) for vs in groups.values())


@pytest.mark.parametrize("conv", ["gcn", "sage", "gat"])
def test_bisimilar_nodes_have_identical_outputs(graph: dict, conv: str) -> None:
    """Corollary 4: equal colours after ``rounds_for(L, conv)`` rounds ⇒ equal model outputs."""
    model = make_model(conv)
    with torch.no_grad():
        out = model(graph["x"], graph["edge_index"])
    rounds = rounds_for(NUM_LAYERS, conv)
    colours = refine(graph["x"], graph["edge_index"], rounds)[rounds]
    assert len(set(colours)) < len(colours)  # there *are* non-trivial equivalence classes
    assert max_within_class_gap(out, colours) < 1e-5


def test_gcn_needs_the_extra_degree_round(graph: dict) -> None:
    """With only L rounds, GCN's neighbour-degree normalisation breaks exactness."""
    model = make_model("gcn")
    with torch.no_grad():
        out = model(graph["x"], graph["edge_index"])
    colours = refine(graph["x"], graph["edge_index"], NUM_LAYERS)[NUM_LAYERS]
    assert max_within_class_gap(out, colours) > 1e-4


def test_local_inference_is_exact(graph: dict) -> None:
    model = make_model("gcn")
    with torch.no_grad():
        full = model(graph["x"], graph["edge_index"])
        for node in (0, 17, 333):
            sub = k_hop_subgraph(
                [node], rounds_for(NUM_LAYERS, "gcn"), graph["x"], graph["edge_index"]
            )
            local = model(sub.x, sub.edge_index)[sub.local_id(node)]
            torch.testing.assert_close(local, full[node])


def _views(graph: dict, conv: str = "sage", n: int = 150) -> list:
    public = graph["train_mask"].nonzero().flatten().tolist()
    return sample_views(
        make_model(conv), graph, public, rounds_for(NUM_LAYERS, conv), num_views=n, seed=0
    )


def test_minimize_collapses_equivalent_nodes_and_keeps_lookups(graph: dict) -> None:
    view = _views(graph, n=1)[0]
    before = list(view.memo)
    minimize(view)
    assert view.memo_entries <= len(before)
    for colour, out in before:
        torch.testing.assert_close(view.lookup(colour), out, atol=1e-5, rtol=0)


def test_greedy_selection_respects_budget_and_prefers_coverage(graph: dict) -> None:
    views = _views(graph)
    node = int(graph["test_mask"].nonzero()[0])
    rounds = rounds_for(NUM_LAYERS, "sage")
    sub = k_hop_subgraph([node], rounds, graph["x"], graph["edge_index"])
    colours = refine(sub.x, sub.edge_index, rounds)
    seed = sub.local_id(node)
    targets = [
        (v, rounds - int(d), colours[rounds - int(d)][v])
        for v, d in enumerate(sub.distance.tolist())
        if d <= NUM_LAYERS
    ]
    seed_colours = [colours[r][seed] for r in range(rounds + 1)]
    chosen = VIEW_SELECTORS.build("gvsel").select(views, seed_colours, targets, 3, 0)
    assert 0 < len(chosen) <= 3
    best_single = max(len(coverage(v, targets)) for v in views)
    assert len(set().union(*(coverage(v, targets) for v in chosen))) >= best_single


def test_service_routes_and_traces(graph: dict) -> None:
    model = make_model("sage")
    views = [minimize(v) for v in _views(graph, n=300)]
    store = InMemoryTraceStore()
    service = GMVService(
        model,
        graph,
        views,
        VIEW_SELECTORS.build("gvsel"),
        NUM_LAYERS,
        rounds_for(NUM_LAYERS, "sage"),
        k=5,
        tracer=Tracer(store),
    )
    with torch.no_grad():
        full = model(graph["x"], graph["edge_index"])
    queries = graph["test_mask"].nonzero().flatten().tolist()[:60]
    answers = [service.answer(n) for n in queries]

    assert sum(service.routes.values()) == len(queries)
    assert service.routes[VIEW_EXACT] > 0 and service.routes[FALLBACK] > 0
    for a in answers:  # both routes are exact when no approximation is allowed
        torch.testing.assert_close(a.logits, full[a.node], atol=1e-5, rtol=0)
    records = store.records()
    assert len(records) == len(queries) and {r.action for r in records} <= {VIEW_EXACT, FALLBACK}


def test_matching_depth_for_layer_specs() -> None:
    """Degree-free layers need L rounds; GCN and unknown/custom layers get the safe L + 1."""
    assert rounds_for(2, "sage") == 2
    assert rounds_for(2, {"name": "gat", "heads": 8}) == 2
    assert rounds_for(2, "gcn") == 3
    assert rounds_for(2, {"name": "my_custom_conv"}) == 3
