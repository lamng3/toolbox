"""The training loop skeleton (pattern: *Template Method*).

:meth:`BaseTrainer.fit` fixes the *order* of operations — setup, resume, step, log, evaluate,
checkpoint, teardown — while subclasses fill in the paradigm-specific parts:

* :meth:`compute_loss` — the loss of one batch (supervised / preference trainers), or
* :meth:`train_step` — a whole custom step (RL trainers that roll out before updating).

``fit`` is decomposed into small, single-purpose methods (pattern: *Refactor a Long Function*),
each of which can be overridden or tested on its own. Every dependency (model, optimizer,
data, callbacks, strategy) is passed in by the caller (pattern: *Dependency Injection*), so a
trainer can be unit-tested with fakes and assembled from config by
:class:`~toolbox.experiment.ExperimentBuilder`.
"""

from __future__ import annotations

import json
from abc import ABC
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

from toolbox.core.constants import RESOLVED_CONFIG_NAME
from toolbox.core.events import Event, EventBus
from toolbox.core.logging import get_logger
from toolbox.core.types import Batch, LossOutput, Metrics, to_device
from toolbox.distributed import SingleDevice, Strategy
from toolbox.trainer.callbacks import Callback
from toolbox.trainer.checkpoint import CheckpointManager
from toolbox.trainer.config import TrainerConfig
from toolbox.trainer.precision import Precision

RESUME_AUTO = "auto"
EVAL_PREFIX = "eval/"

log = get_logger("trainer")


class BaseTrainer(ABC):
    """Shared training loop. Subclasses implement :meth:`compute_loss` or :meth:`train_step`."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
        train_data: Dataset[Any] | None = None,
        eval_data: Dataset[Any] | None = None,
        collate_fn: Any = None,
        scheduler: LRScheduler | None = None,
        callbacks: Sequence[Callback] = (),
        strategy: Strategy | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy or SingleDevice(config.device)
        self.strategy.setup()
        self.device = self.strategy.device
        self.model = self.strategy.wrap_model(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data, self.eval_data, self.collate_fn = train_data, eval_data, collate_fn
        self.precision = Precision(config.precision, self.device)
        self.checkpoints = CheckpointManager(config.output_dir, config.keep_last)
        self.run_config = run_config or {}
        self.events = EventBus()
        self.callbacks = list(callbacks)
        for callback in self.callbacks:
            callback.attach(self.events)
        self.step = 0
        self.last_metrics: Metrics = {}
        self._stop_reason: str | None = None
        self._train_iter: Iterator[Any] | None = None

    # ------------------------------------------------------------------ template method --
    def fit(self) -> Metrics:
        """Run training to ``max_steps`` (or until a callback requests a stop)."""
        self._prepare_run()
        try:
            self.events.emit(Event.FIT_START, trainer=self)
            while not self._should_stop():
                self._run_one_step()
                self._maybe_evaluate()
                self._maybe_save()
            self._finish()
        except BaseException as error:
            self.events.emit(Event.EXCEPTION, trainer=self, error=error)
            raise
        finally:
            self.strategy.teardown()
        return self.last_metrics

    # ------------------------------------------------------------------- hooks to override --
    def setup(self) -> None:
        """Hook: called once before the first step (after resume)."""

    def compute_loss(self, batch: Batch) -> LossOutput:
        """Hook: loss of one batch. Required unless :meth:`train_step` is overridden."""
        raise NotImplementedError(f"{type(self).__name__} must implement compute_loss")

    def train_step(self) -> Metrics:
        """Hook: one optimizer step. Default = gradient accumulation over ``compute_loss``."""
        sums: dict[str, float] = defaultdict(float)
        self.optimizer.zero_grad(set_to_none=True)
        accum = self.config.grad_accum_steps
        for _ in range(accum):
            batch = self.next_batch()
            with self.precision.autocast():
                out = self.compute_loss(batch)
            self.precision.backward(out.loss / accum)
            for key, value in out.detached_metrics().items():
                sums[key] += value / accum
            if "input_ids" in batch:
                sums["tokens"] += float(batch["input_ids"].numel())
        sums["grad_norm"] = self.optimizer_step()
        return dict(sums)

    @torch.no_grad()
    def evaluate(self) -> Metrics:
        """Hook: average ``compute_loss`` metrics over the eval set."""
        if self.eval_data is None:
            return {}
        self.model.eval()
        sums: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in self._loader(self.eval_data, shuffle=False):
            if self.config.eval_batches and batches >= self.config.eval_batches:
                break
            with self.precision.autocast():
                out = self.compute_loss(to_device(batch, self.device))
            for key, value in out.detached_metrics().items():
                sums[key] += value
            batches += 1
        self.model.train()
        return {
            f"{EVAL_PREFIX}{k}": self.strategy.reduce_mean(v / max(batches, 1))
            for k, v in sums.items()
        }

    def state_dict(self) -> dict[str, Any]:
        """Hook: everything needed to resume. Extend in subclasses with extra state."""
        return {
            "model": self.unwrapped_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "step": self.step,
            "config": self.run_config,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.unwrapped_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        self.step = int(state["step"])

    # --------------------------------------------------------------------- building blocks --
    @property
    def unwrapped_model(self) -> nn.Module:
        return self.strategy.unwrap_model(self.model)

    @property
    def is_main_process(self) -> bool:
        return self.strategy.is_main_process

    def request_stop(self, reason: str) -> None:
        """Ask the loop to stop after the current step (used by callbacks)."""
        self._stop_reason = reason

    def next_batch(self) -> Batch:
        """Next training batch on the right device; cycles over the dataset forever."""
        if self.train_data is None:
            raise RuntimeError(f"{type(self).__name__} has no training data")
        if self._train_iter is None:
            self._train_iter = self._infinite(self._loader(self.train_data, shuffle=True))
        batch = next(self._train_iter)
        return to_device(batch, self.device) if isinstance(batch, dict) else batch

    def optimizer_step(self, loss: torch.Tensor | None = None) -> float:
        """Optionally backprop ``loss``, then clip, step optimizer + scheduler, zero grads."""
        if loss is not None:
            self.precision.backward(loss)
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = self.precision.clip_and_step(self.optimizer, params, self.config.grad_clip)
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    def save_checkpoint(self) -> Path | None:
        path = None
        if self.is_main_process:
            path = self.checkpoints.save(self.state_dict(), self.step)
            self.events.emit(Event.CHECKPOINT_SAVED, trainer=self, path=path)
        self.strategy.barrier()
        return path

    # ------------------------------------------------------------------------- internals --
    def _prepare_run(self) -> None:
        self._write_resolved_config()
        self._maybe_resume()
        self.model.train()
        self.setup()

    def _run_one_step(self) -> None:
        self.events.emit(Event.STEP_START, trainer=self, step=self.step + 1)
        metrics = self.train_step()
        self.step += 1
        if self.scheduler:
            metrics["lr"] = float(self.scheduler.get_last_lr()[0])
        self.events.emit(Event.STEP_END, trainer=self, step=self.step, metrics=metrics)
        self.last_metrics = metrics

    def _maybe_evaluate(self) -> None:
        every = self.config.eval_every
        if every and self.step % every == 0:
            self._evaluate_and_emit()

    def _evaluate_and_emit(self) -> None:
        metrics = self.evaluate()
        if metrics:
            self.events.emit(Event.EVAL_END, trainer=self, step=self.step, metrics=metrics)
            self.last_metrics = {**self.last_metrics, **metrics}

    def _maybe_save(self) -> None:
        every = self.config.save_every
        if every and self.step % every == 0:
            self.save_checkpoint()

    def _finish(self) -> None:
        if self._stop_reason:
            log.info("Stopping early: %s", self._stop_reason)
        # Skip the final eval/save when the periodic one just ran at this exact step.
        if self.eval_data is not None and not self._ran_this_step(self.config.eval_every):
            self._evaluate_and_emit()
        if not self._ran_this_step(self.config.save_every):
            self.save_checkpoint()
        self.events.emit(Event.FIT_END, trainer=self, metrics=self.last_metrics)

    def _ran_this_step(self, every: int) -> bool:
        return bool(every) and self.step > 0 and self.step % every == 0

    def _should_stop(self) -> bool:
        return self.step >= self.config.max_steps or self._stop_reason is not None

    def _maybe_resume(self) -> None:
        source = self.config.resume_from
        if not source:
            return
        path = self.checkpoints.latest() if source == RESUME_AUTO else Path(source)
        if path is None or not path.exists():
            log.info("No checkpoint to resume from (%s); starting fresh", source)
            return
        self.load_state_dict(self.checkpoints.load(path, map_location=self.device))
        log.info("Resumed from %s at step %d", path, self.step)

    def _write_resolved_config(self) -> None:
        if not self.is_main_process or not self.run_config:
            return
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            from omegaconf import OmegaConf

            OmegaConf.save(OmegaConf.create(self.run_config), out / RESOLVED_CONFIG_NAME)
        except ImportError:  # pragma: no cover - omegaconf is a core dependency
            (out / RESOLVED_CONFIG_NAME).write_text(json.dumps(self.run_config, indent=2))

    def _loader(self, dataset: Dataset[Any], shuffle: bool) -> DataLoader[Any]:
        return self.strategy.dataloader(
            dataset, self.config.batch_size, shuffle, self.collate_fn, self.config.num_workers
        )

    @staticmethod
    def _infinite(loader: DataLoader[Any]) -> Iterator[Any]:
        epoch = 0
        while True:
            sampler: Any = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)  # reshuffle differently each epoch under DDP
            yield from loader
            epoch += 1
