"""Reproduce the GMV experiment design with toolbox.

    python -m examples.gmv.run                                  # defaults: GCN, 400 molecules
    python -m examples.gmv.run model.conv=gat --views 500 --k 5  # any toolbox override works
    python -m examples.gmv.run 'model.head={name:mlp,hidden_dims:[32]}' model.norm=layernorm
    python -m examples.gmv.run model.conv=pyg_gcn               # PyG's GCNConv (needs .[gnn])

Steps:
  1. Train a 2-layer GNN with the toolbox trainer (config ``experiment=gmv_molecules_gcn``).
  2. Offline: materialise views over subgraphs of *public* molecules.
  3. Online: answer node queries on *private* molecules with each variant and compare against
     full-graph inference (fidelity = same prediction as the full graph).

The variants mirror the upstream scripts (Yangxin666/GMV ``Codes/``):
  GVInf        forgetting-weighted GVSel + GVMin
  GVInf-all    plain greedy GVSel + GVMin
  GVInf-ran    random view selection + GVMin
  GVInf-nomin  greedy GVSel without GVMin
  local        no views: always run the model on G^{L+1}(v_t) (exact baseline)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from examples.gmv.bisim import conv_name, model_conv, rounds_for
from examples.gmv.selection import VIEW_SELECTORS
from examples.gmv.service import FALLBACK, VIEW_APPROX, VIEW_EXACT, GMVService
from examples.gmv.views import GraphModelView, minimize, sample_views
from toolbox.experiment import ExperimentBuilder, load_config
from toolbox.tracing import JsonlTraceStore, Tracer

HERE = Path(__file__).parent
EXPERIMENT = "gmv_molecules_gcn"
MS_PER_S = 1000.0
BYTES_PER_KB = 1024


@dataclass(frozen=True)
class Variant:
    name: str
    selector: str | None  # None → no views (local inference baseline)
    minimized: bool


VARIANTS = (
    Variant("GVInf", "gvsel_forgetting", True),
    Variant("GVInf-all", "gvsel", True),
    Variant("GVInf-ran", "random", True),
    Variant("GVInf-nomin", "gvsel", False),
    Variant("local", None, False),
)


def train_model(overrides: list[str], out: Path) -> tuple[torch.nn.Module, dict, int, Any]:
    cfg = load_config(
        [f"experiment={EXPERIMENT}", f"output_dir={out / 'train'}", *overrides],
        config_dir=HERE / "configs",
    )
    builder = ExperimentBuilder.from_config(cfg)
    trainer = builder.build()
    trainer.fit()
    graph = builder.data().train[0]
    model = trainer.unwrapped_model.cpu().eval()
    conv = model_conv(builder.run_config["model"])  # e.g. "gcn" or {"name": "gat", ...}
    return model, graph, int(cfg.model.num_layers), conv


def run_variant(
    variant: Variant,
    model: torch.nn.Module,
    graph: dict,
    base_views: list,
    num_layers: int,
    rounds: int,
    queries: list[int],
    full_logits: torch.Tensor,
    args: argparse.Namespace,
    tracer: Tracer,
) -> dict:
    views: list[GraphModelView] = copy.deepcopy(base_views) if variant.selector else []
    if variant.minimized:
        views = [minimize(v) for v in views]
    selector = VIEW_SELECTORS.build(variant.selector or "random")
    service = GMVService(
        model,
        graph,
        views,
        selector,
        num_layers,
        rounds,
        k=args.k,
        min_rounds=args.min_rounds,
        tracer=tracer,
        agent=variant.name,
    )
    answers = [service.answer(n) for n in queries]

    preds = torch.stack([a.logits for a in answers]).argmax(-1)
    exact = [a for a in answers if a.route == VIEW_EXACT]
    exact_err = max(
        (float((a.logits - full_logits[a.node]).abs().max()) for a in exact), default=0.0
    )
    n = len(answers)
    return {
        "variant": variant.name,
        "fidelity": float((preds == full_logits[queries].argmax(-1)).float().mean()),
        "accuracy": float((preds == graph["y"][queries]).float().mean()),
        "view_exact": service.routes[VIEW_EXACT] / n,
        "view_approx": service.routes[VIEW_APPROX] / n,
        "fallback": service.routes[FALLBACK] / n,
        "model_calls_avoided": 1 - service.routes[FALLBACK] / n,
        "max_abs_err_exact": exact_err,
        "mean_latency_ms": sum(a.latency_ms for a in answers) / n,
        "memo_entries": sum(v.memo_entries for v in views),
        "memo_kb": sum(v.memo_bytes() for v in views) / BYTES_PER_KB,
    }


def print_table(rows: list[dict], full: dict) -> None:
    table = Table(
        title=f"GMV ({full['conv']}, {full['rounds']} rounds) on "
        f"{full['num_queries']} private-molecule queries "
        f"(full-graph accuracy {full['accuracy']:.3f}, "
        f"one full pass {full['time_ms']:.1f} ms)"
    )
    cols = [
        "variant",
        "fidelity",
        "accuracy",
        "view_exact",
        "view_approx",
        "fallback",
        "max_abs_err_exact",
        "mean_latency_ms",
        "memo_entries",
        "memo_kb",
    ]
    for c in cols:
        table.add_column(c, justify="left" if c == "variant" else "right")
    for row in rows:
        table.add_row(
            *[
                row[c]
                if isinstance(row[c], str)
                else (f"{row[c]:.3g}" if isinstance(row[c], float) else str(row[c]))
                for c in cols
            ]
        )
    Console(width=200).print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("overrides", nargs="*", help="toolbox/Hydra overrides for training")
    parser.add_argument("--out", type=Path, default=Path("runs/gmv"))
    parser.add_argument("--views", type=int, default=300, help="number of materialised views")
    parser.add_argument("--k", type=int, default=5, help="view budget per query (GVSel)")
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=None,
        help="lowest bisimulation depth accepted for approximate answers "
        "(default: exact answers only)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, graph, num_layers, conv = train_model(args.overrides, args.out)
    rounds = rounds_for(num_layers, conv)

    start = time.perf_counter()
    with torch.no_grad():
        full_logits = model(graph["x"], graph["edge_index"])
    full_ms = (time.perf_counter() - start) * MS_PER_S
    private = graph["test_mask"].nonzero().flatten().tolist()
    public = graph["train_mask"].nonzero().flatten().tolist()
    queries = random.Random(args.seed).sample(private, min(args.queries, len(private)))
    full = {
        "num_queries": len(queries),
        "conv": conv_name(conv),
        "rounds": rounds,
        "time_ms": full_ms,
        "accuracy": float((full_logits[queries].argmax(-1) == graph["y"][queries]).float().mean()),
    }

    base_views = sample_views(model, graph, public, rounds, num_views=args.views, seed=args.seed)
    tracer = Tracer(JsonlTraceStore(str(args.out / "traces")))
    rows = [
        run_variant(
            v, model, graph, base_views, num_layers, rounds, queries, full_logits, args, tracer
        )
        for v in VARIANTS
    ]

    print_table(rows, full)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps({"full_graph": full, "variants": rows}, indent=2)
    )
    print(f"results → {args.out / 'results.json'}   traces → toolbox traces {args.out / 'traces'}")


if __name__ == "__main__":
    main()
