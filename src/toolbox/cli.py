"""``toolbox`` command-line interface.

Every sub-command builds a command/config object and delegates — the CLI holds no training
logic. Overrides use Hydra syntax: ``toolbox train experiment=grpo_tiny trainer.max_steps=20``.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="toolbox",
    help="An extensible toolbox for machine learning research: train, launch and serve.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()

Overrides = Annotated[
    list[str] | None, typer.Argument(help="Hydra overrides, e.g. experiment=sft_tiny lr=1e-4")
]
ConfigDir = Annotated[
    str | None, typer.Option("--config-dir", help="Extra directory of configs to search first")
]


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    result = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(result, dict)
    return result  # type: ignore[return-value]


@app.command()
def train(overrides: Overrides = None, config_dir: ConfigDir = None) -> None:
    """Train in this process (works under torchrun / srun too)."""
    from toolbox.experiment import load_config, run_experiment

    metrics = run_experiment(load_config(overrides or [], config_dir))
    console.print_json(json.dumps(metrics))


@app.command()
def launch(
    overrides: Overrides = None,
    config_dir: ConfigDir = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print, don't run")] = False,
) -> None:
    """Launch with the configured launcher (``launcher=local|torchrun|slurm``)."""
    from toolbox.core.registry import LAUNCHERS
    from toolbox.experiment import load_config
    from toolbox.launch import CommandInvoker

    overrides = overrides or []
    cfg = _cfg_to_dict(load_config(overrides, config_dir))
    spec = dict(cfg["launcher"])
    passthrough = [o for o in overrides if not o.startswith("launcher")]
    inject = {"overrides": passthrough, "output_dir": cfg["output_dir"], "config_dir": config_dir}
    command = LAUNCHERS.build(spec, inject=inject)
    result = CommandInvoker().run(command, dry_run=dry_run)
    console.print(result if isinstance(result, str) else result.detail)


@app.command("config")
def show_config(overrides: Overrides = None, config_dir: ConfigDir = None) -> None:
    """Print the fully composed config."""
    from omegaconf import OmegaConf

    from toolbox.experiment import load_config

    console.print(OmegaConf.to_yaml(load_config(overrides or [], config_dir), resolve=True))


@app.command("list")
def list_components(
    kind: Annotated[str | None, typer.Argument(help="Registry, e.g. objective")] = None,
) -> None:
    """List registered components (built-ins and plugins)."""
    from toolbox.core.registry import ALL_REGISTRIES, load_builtins

    load_builtins()
    table = Table("registry", "components")
    for name, registry in sorted(ALL_REGISTRIES.items()):
        if kind is None or kind == name:
            table.add_row(name, ", ".join(registry.names()))
    console.print(table)


@app.command()
def env() -> None:
    """Print the software/hardware environment (handy on a fresh VM or compute node)."""
    from toolbox.core.env import collect_env

    console.print_json(json.dumps(collect_env(), default=str))


@app.command()
def serve(
    engine: Annotated[str, typer.Option(help="torch | hf | sglang | vllm")] = "torch",
    model: Annotated[str | None, typer.Option(help="HF id/path (hf, sglang, vllm)")] = None,
    checkpoint: Annotated[str | None, typer.Option(help="toolbox checkpoint (torch)")] = None,
    host: str = "127.0.0.1",
    port: int = 30000,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Serve a model behind an OpenAI-compatible ``/v1/completions`` endpoint."""
    import subprocess

    from toolbox.serving import ServerLaunchSpec, serve_engine

    if engine in ("sglang", "vllm"):
        if not model:
            raise typer.BadParameter("--model is required for sglang/vllm")
        spec = ServerLaunchSpec(engine, model, host, port)
        console.print(spec.shell())
        if not dry_run:
            subprocess.run(spec.command(), check=True)
        return
    backend = _local_engine(engine, model, checkpoint)
    if dry_run:
        console.print(f"would serve {backend.name} on http://{host}:{port}")
        return
    server = serve_engine(backend, host, port)
    console.print(f"Serving {backend.name} on http://{host}:{port}/v1/completions")
    server.serve_forever()


def _local_engine(engine: str, model: str | None, checkpoint: str | None) -> Any:
    from toolbox.core.registry import ENGINES

    if engine == "hf":
        if not model:
            raise typer.BadParameter("--model is required for hf")
        return ENGINES.build({"name": "hf", "model_name_or_path": model})
    if checkpoint:
        from toolbox.trainer.checkpoint import CheckpointManager

        state = CheckpointManager(".").load(checkpoint)
        model_spec = state.get("config", {}).get("model")
        tokenizer_spec = state.get("config", {}).get("tokenizer", {"name": "byte"})
        if model_spec is None:
            raise typer.BadParameter("checkpoint has no model config; pass --engine hf")
        return ENGINES.build(
            {
                "name": "torch",
                "model": model_spec,
                "tokenizer": tokenizer_spec,
                "checkpoint": checkpoint,
            }
        )
    return ENGINES.build({"name": "torch", "model": {"name": "tiny_gpt"}, "tokenizer": "byte"})


@app.command()
def bench(
    engine: Annotated[str, typer.Option(help="torch | hf | sglang | vllm | openai_http")] = "torch",
    model: Annotated[str | None, typer.Option()] = None,
    base_url: Annotated[str, typer.Option()] = "http://127.0.0.1:30000/v1",
    num_prompts: int = 32,
    batch_size: int = 8,
    max_new_tokens: int = 32,
) -> None:
    """Measure latency/throughput of an inference engine."""
    from toolbox.core.registry import ENGINES
    from toolbox.data.repository import load_records
    from toolbox.serving import SamplingParams, run_benchmark

    if engine == "openai_http":
        backend = ENGINES.build(
            {"name": "openai_http", "base_url": base_url, "model": model or "default"}
        )
    elif engine == "sglang":
        backend = ENGINES.build({"name": "sglang", "model_path": model})
    elif engine == "vllm":
        backend = ENGINES.build({"name": "vllm", "model": model})
    else:
        backend = _local_engine(engine, model, None)
    prompts = [r["prompt"] for r in load_records("builtin://arithmetic.jsonl")][:num_prompts]
    params = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0)
    result = run_benchmark(backend, prompts, params, batch_size)
    console.print_json(json.dumps(result.as_dict()))


@app.command()
def traces(path: Annotated[str, typer.Argument(help="Directory with actions.jsonl")]) -> None:
    """Summarise an agent action trace log (counts, errors, rewards, latency)."""
    from toolbox.tracing import JsonlTraceStore, summarize

    table = Table("agent/kind", "count", "errors", "rewarded", "mean reward", "mean latency ms")
    for key, row in summarize(JsonlTraceStore(path)).items():
        table.add_row(
            key,
            *[
                "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
                for v in row.values()
            ],
        )
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
