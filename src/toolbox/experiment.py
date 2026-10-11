"""Assemble and run experiments (pattern: *Builder*).

An experiment is a set of named component specs — tokenizer, data, model, objective,
optimizer, scheduler, trainer, reward, advantage, env, strategy, callbacks. The
:class:`ExperimentBuilder` collects them (fluently from Python, or from a Hydra config), then
builds each component through its registry (*Factory*) and wires dependencies between them
(*Dependency Injection*): the tokenizer goes to the dataset, data-derived shapes go to the
model, the model's parameters go to the optimizer, and the finished parts go to the trainer.

From Python::

    trainer = (
        ExperimentBuilder("sft-demo")
        .with_tokenizer("byte")
        .with_data("sft", max_length=64)
        .with_model("tiny_gpt", d_model=64, n_layers=2)
        .with_objective("sft")
        .with_trainer("supervised", max_steps=50)
        .build()
    )
    trainer.fit()

From the shell::

    toolbox train experiment=sft_tiny trainer.max_steps=50
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from toolbox.core.constants import DEFAULT_SEED
from toolbox.core.env import seed_everything
from toolbox.core.logging import RunContext
from toolbox.core.registry import (
    ADVANTAGES,
    CALLBACKS,
    DATASETS,
    ENVS,
    MODELS,
    OBJECTIVES,
    OPTIMIZERS,
    REWARDS,
    SCHEDULERS,
    STRATEGIES,
    TOKENIZERS,
    TRAINERS,
)
from toolbox.core.types import Metrics

Spec = dict[str, Any]

CONFIG_DIR = Path(__file__).parent / "configs"
PRIMARY_CONFIG = "config"
COMPONENT_KEYS = (
    "tokenizer",
    "data",
    "model",
    "objective",
    "optimizer",
    "scheduler",
    "trainer",
    "reward",
    "advantage",
    "env",
    "strategy",
)
DEFAULT_CALLBACKS: tuple[Spec, ...] = ({"name": "console"}, {"name": "jsonl"})
INIT_FROM_KEY = "init_from"
_MISSING = object()


def load_weights(model: Any, checkpoint: str | Path) -> None:
    """Load ``model`` weights from a toolbox checkpoint (or a raw state dict file)."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)


def _spec(name: str, kwargs: Mapping[str, Any]) -> Spec:
    return {"name": name, **kwargs}


class ExperimentBuilder:
    """Collects component specs, then builds a ready-to-``fit`` trainer."""

    def __init__(
        self, name: str = "experiment", seed: int = DEFAULT_SEED, output_dir: str | None = None
    ) -> None:
        self.name, self.seed = name, seed
        self.output_dir = output_dir or f"runs/{name}"
        self.specs: dict[str, Spec | None] = {key: None for key in COMPONENT_KEYS}
        self.specs["optimizer"] = {"name": "adamw"}
        self.specs["scheduler"] = {"name": "constant"}
        self.specs["strategy"] = {"name": "auto"}
        self.callback_specs: list[Spec] = list(DEFAULT_CALLBACKS)
        self.extra: dict[str, Any] = {}
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------- fluent setters --
    def with_tokenizer(self, name: str, **kwargs: Any) -> ExperimentBuilder:
        return self._set("tokenizer", name, kwargs)

    def with_data(self, name: str, **kwargs: Any) -> ExperimentBuilder:
        return self._set("data", name, kwargs)

    def with_model(self, name: str, **kwargs: Any) -> ExperimentBuilder:
        return self._set("model", name, kwargs)

    def with_objective(self, name: str, **kwargs: Any) -> ExperimentBuilder:
        return self._set("objective", name, kwargs)

    def with_optimizer(self, name: str, **kwargs: Any) -> ExperimentBuilder:
        return self._set("optimizer", name, kwargs)

    def with_scheduler(self, name: str, **kwargs: Any) -> ExperimentBuilder:
        return self._set("scheduler", name, kwargs)

    def with_trainer(self, name: str, **kwargs: Any) -> ExperimentBuilder:
        return self._set("trainer", name, kwargs)

    def with_reward(self, name: str, **kwargs: Any) -> ExperimentBuilder:
        return self._set("reward", name, kwargs)

    def with_advantage(self, name: str, **kwargs: Any) -> ExperimentBuilder:
        return self._set("advantage", name, kwargs)

    def with_env(self, name: str, **kwargs: Any) -> ExperimentBuilder:
        return self._set("env", name, kwargs)

    def with_strategy(self, name: str, **kwargs: Any) -> ExperimentBuilder:
        return self._set("strategy", name, kwargs)

    def with_callbacks(self, *specs: str | Spec) -> ExperimentBuilder:
        self.callback_specs = [{"name": s} if isinstance(s, str) else dict(s) for s in specs]
        return self

    def _set(self, key: str, name: str, kwargs: Mapping[str, Any]) -> ExperimentBuilder:
        self.specs[key] = _spec(name, kwargs)
        self._cache.clear()
        return self

    @classmethod
    def from_config(cls, cfg: Any) -> ExperimentBuilder:
        """Create a builder from a (Hydra/OmegaConf or plain) config mapping.

        Modules listed under ``imports`` are imported first, so components they register
        (a dataset, a loss, a layer in your project) can be selected by name from the CLI,
        ``toolbox launch`` and SLURM jobs without packaging them as a plugin.
        """
        data = _to_container(cfg)
        import_modules(data.get("imports") or [])
        builder = cls(
            data.get("name", "experiment"), data.get("seed", DEFAULT_SEED), data.get("output_dir")
        )
        for key in COMPONENT_KEYS:
            if data.get(key) is not None:
                builder.specs[key] = dict(data[key])
        if data.get("callbacks") is not None:
            builder.callback_specs = [dict(c) for c in data["callbacks"]]
        builder.extra = {
            k: v
            for k, v in data.items()
            if k not in (*COMPONENT_KEYS, "callbacks", "imports", "name", "seed", "output_dir")
        }
        return builder

    # --------------------------------------------------------- component accessors (cached) --
    def _cached(self, key: str, factory: Any) -> Any:
        value = self._cache.get(key, _MISSING)
        if value is _MISSING:
            value = factory()
            self._cache[key] = value
        return value

    def _require(self, key: str) -> Spec:
        spec = self.specs.get(key)
        if spec is None:
            raise ValueError(f"Experiment '{self.name}' needs a '{key}' section")
        return spec

    def tokenizer(self) -> Any:
        spec = self.specs["tokenizer"]
        return self._cached("tokenizer", lambda: TOKENIZERS.build(spec) if spec else None)

    def data(self) -> Any:
        spec = self.specs["data"]
        inject = {"tokenizer": self.tokenizer(), "seed": self.seed}
        return self._cached("data", lambda: DATASETS.build(spec, inject=inject) if spec else None)

    def env(self) -> Any:
        spec = self.specs["env"]
        return self._cached("env", lambda: ENVS.build(spec) if spec else None)

    def model_inject(self) -> dict[str, Any]:
        """Facts the model may need but the config should not have to repeat."""
        inject: dict[str, Any] = {}
        if (tok := self.tokenizer()) is not None:
            inject["vocab_size"] = tok.vocab_size
        if (data := self.data()) is not None:
            inject.update(data.info)
        if (env := self.env()) is not None:
            inject.update(obs_dim=env.obs_dim, num_actions=env.num_actions)
        return inject

    def model(self) -> Any:
        """Build the model; ``init_from: <checkpoint>`` in the spec loads weights from a
        previous stage (SFT → DPO → GRPO pipelines)."""
        spec = dict(self._require("model"))
        init_from = spec.pop(INIT_FROM_KEY, None)

        def build() -> Any:
            model = MODELS.build(spec, inject=self.model_inject())
            if init_from:
                load_weights(model, init_from)
            return model

        return self._cached("model", build)

    def objective(self) -> Any:
        spec = self._require("objective")
        return self._cached("objective", lambda: OBJECTIVES.build(spec))

    def reward(self) -> Any:
        spec = self._require("reward")
        return self._cached(
            "reward", lambda: REWARDS.build(spec, inject={"tokenizer": self.tokenizer()})
        )

    def advantage(self) -> Any:
        spec = self.specs["advantage"] or {"name": "group_relative"}
        return self._cached("advantage", lambda: ADVANTAGES.build(spec))

    def optimizer(self, model: Any) -> Any:
        return OPTIMIZERS.build(self._require("optimizer"), model=model)

    def scheduler(self, optimizer: Any, max_steps: int) -> Any:
        spec = self.specs["scheduler"]
        return SCHEDULERS.build(spec, optimizer=optimizer, max_steps=max_steps) if spec else None

    def strategy(self, device: str) -> Any:
        return STRATEGIES.build(self._require("strategy"), inject={"device": device})

    def callbacks(self) -> list[Any]:
        return [CALLBACKS.build(spec) for spec in self.callback_specs]

    @property
    def run_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "output_dir": self.output_dir,
            **{k: v for k, v in self.specs.items() if v is not None},
            "callbacks": self.callback_specs,
            **self.extra,
        }

    # ------------------------------------------------------------------------- build --
    def build(self) -> Any:
        """Seed, configure the run context, and return a ready trainer."""
        from toolbox.trainer.config import TrainerConfig

        seed_everything(self.seed)
        RunContext.configure(self.name, self.output_dir)
        trainer_spec = dict(self._require("trainer"))
        trainer_cls: Any = TRAINERS.get(trainer_spec.pop("name"))
        trainer_spec.setdefault("output_dir", self.output_dir)
        config, trainer_kwargs = TrainerConfig.split(trainer_spec)
        return trainer_cls.from_builder(self, config, **trainer_kwargs)


def _to_container(cfg: Any) -> dict[str, Any]:
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            container = OmegaConf.to_container(cfg, resolve=True)
            assert isinstance(container, dict)
            return {str(k): v for k, v in container.items()}
    except ImportError:  # pragma: no cover
        pass
    return dict(cfg)


def load_config(overrides: Sequence[str] = (), config_dir: str | Path | None = None) -> Any:
    """Compose the Hydra config from ``configs/`` plus command-line style overrides.

    ``config_dir`` adds a user directory searched *before* the packaged configs, so a project
    can add its own ``experiment/*.yaml`` without forking the toolbox.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    overrides = list(overrides)
    if config_dir is not None:
        overrides.append(f"hydra.searchpath=[file://{Path(config_dir).resolve()}]")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=PRIMARY_CONFIG, overrides=overrides)


def import_modules(modules: Sequence[str]) -> None:
    """Import modules that register components, e.g. ``examples.jev.data``.

    The current directory is searched too (as with ``python -m``), so modules of the project
    you run from are found by the ``toolbox`` command.
    """
    if modules and str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    for module in modules:
        importlib.import_module(module)


def run_experiment(cfg: Any) -> Metrics:
    """Build the trainer described by ``cfg`` and train it. Returns the final metrics."""
    trainer = ExperimentBuilder.from_config(cfg).build()
    metrics: Metrics = trainer.fit()
    return metrics
