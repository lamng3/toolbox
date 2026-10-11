# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this is

`toolbox` (import `toolbox`, CLI `toolbox`) is a post-training / RL library and template. The
focus is SFT, DPO, GRPO-family RL, decision models, agent tracing, and serving. Pretraining is
only a minimal example. The code also teaches design patterns (see
`docs/concepts/design-patterns.md`), so keep them clean when you change code.

## Commands

```bash
export PYTHONPATH=src            # macOS hides .pth files in .venv; the Makefile sets this
make test-fast                   # unit tests
make test                        # + end-to-end tiny experiments (~1 min)
make lint typecheck              # ruff + mypy (both must pass)
make docs                        # mkdocs build --strict
.venv/bin/python -m toolbox.cli train experiment=sft_tiny trainer.device=cpu
```

## Architecture rules

- Components are registered by name (`core/registry.py`) and built from `{name: ..., **kwargs}`
  specs. Never put import paths in configs.
- `ExperimentBuilder` (`experiment.py`) wires dependencies. Trainers receive objects and never
  construct models or datasets. Optional deps go through `Registry.build(inject=...)`, which
  only passes kwargs the factory accepts.
- `BaseTrainer.fit` is a template method. New paradigms override `compute_loss` or
  `train_step` and add `from_builder`.
- Side effects (logging, saving, tracing, memory) go in callbacks subscribed to `EventBus`
  events, not in loops.
- No magic numbers: use `core/constants.py` or config values.
- Optional heavy deps (transformers, sglang, vllm, PyG, wandb) are imported lazily with
  `core.decorators.require(module, extra)`.
- Configs live in `src/toolbox/configs` (shipped with the package). Each tiny experiment must
  stay CPU-runnable in seconds and be listed in `tests/integration/test_experiments.py`.

## Conventions

Python ≥ 3.10, ruff line length 100, Google-style docstrings (they render in the API docs).
License is CC BY 4.0. After finishing a piece of work: run checks, commit, and push.
