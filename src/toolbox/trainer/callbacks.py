"""Training callbacks (pattern: *Observer*).

A :class:`Callback` subscribes its ``on_*`` methods to the trainer's
:class:`~toolbox.core.events.EventBus`. Logging, early stopping, profiling and memory writing
are all callbacks, so none of them lives inside the training loop.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from toolbox.core.constants import METRICS_FILENAME
from toolbox.core.decorators import require
from toolbox.core.events import Event, EventBus
from toolbox.core.logging import get_logger
from toolbox.core.registry import CALLBACKS

if TYPE_CHECKING:
    from toolbox.trainer.base import BaseTrainer

BYTES_PER_MB = 1024**2
# ru_maxrss is reported in bytes on macOS but kilobytes on Linux.
RSS_TO_MB = 1 / BYTES_PER_MB if sys.platform == "darwin" else 1 / 1024


class Callback:
    """Base class; override any subset of the hooks."""

    def attach(self, bus: EventBus) -> None:
        for event in Event:
            handler = getattr(self, f"on_{event.value}", None)
            if handler is not None and _is_overridden(self, f"on_{event.value}"):
                bus.subscribe(event, handler)

    def on_fit_start(self, trainer: BaseTrainer) -> None: ...
    def on_step_start(self, trainer: BaseTrainer, step: int) -> None: ...
    def on_step_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None: ...
    def on_eval_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None: ...
    def on_checkpoint_saved(self, trainer: BaseTrainer, path: Path) -> None: ...
    def on_rollout_end(self, trainer: BaseTrainer, rollout: Any, rewards: torch.Tensor) -> None: ...
    def on_fit_end(self, trainer: BaseTrainer, metrics: dict[str, float]) -> None: ...
    def on_exception(self, trainer: BaseTrainer, error: BaseException) -> None: ...


def _is_overridden(obj: object, name: str) -> bool:
    return getattr(type(obj), name) is not getattr(Callback, name)


def _format(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items()))


@CALLBACKS.register("console")
class ConsoleLogger(Callback):
    """Prints metrics every ``log_every`` steps (rank 0 only)."""

    def __init__(self, log_every: int = 10) -> None:
        self.log_every = log_every
        self.log = get_logger("train")

    def on_fit_start(self, trainer: BaseTrainer) -> None:
        if trainer.is_main_process:
            self.log.info(
                "Starting %s for %d steps → %s",
                type(trainer).__name__,
                trainer.config.max_steps,
                trainer.config.output_dir,
            )

    def on_step_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        if trainer.is_main_process and self.log_every and step % self.log_every == 0:
            self.log.info("step %d  %s", step, _format(metrics))

    def on_eval_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        if trainer.is_main_process:
            self.log.info("eval @ %d  %s", step, _format(metrics))

    def on_fit_end(self, trainer: BaseTrainer, metrics: dict[str, float]) -> None:
        if trainer.is_main_process:
            self.log.info("Finished at step %d", trainer.step)


@CALLBACKS.register("jsonl")
class JsonlLogger(Callback):
    """Appends ``{"step", "split", **metrics}`` lines to ``<output_dir>/metrics.jsonl``."""

    def __init__(self, filename: str = METRICS_FILENAME) -> None:
        self.filename = filename

    def _write(
        self, trainer: BaseTrainer, split: str, step: int, metrics: dict[str, float]
    ) -> None:
        if not trainer.is_main_process:
            return
        path = Path(trainer.config.output_dir) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"step": step, "split": split, "time": time.time(), **metrics}) + "\n"
            )

    def on_step_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        self._write(trainer, "train", step, metrics)

    def on_eval_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        self._write(trainer, "eval", step, metrics)


@CALLBACKS.register("tensorboard")
class TensorBoardLogger(Callback):
    def __init__(self, log_dir: str | None = None) -> None:
        self.log_dir = log_dir
        self.writer: Any = None

    def on_fit_start(self, trainer: BaseTrainer) -> None:
        if trainer.is_main_process:
            tb = require("torch.utils.tensorboard", extra="tracking")
            self.writer = tb.SummaryWriter(self.log_dir or f"{trainer.config.output_dir}/tb")

    def on_step_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"train/{k}", v, step)

    def on_eval_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)

    def on_fit_end(self, trainer: BaseTrainer, metrics: dict[str, float]) -> None:
        if self.writer:
            self.writer.close()


@CALLBACKS.register("wandb")
class WandbLogger(Callback):
    def __init__(
        self,
        project: str = "toolbox",
        entity: str | None = None,
        mode: str = "online",
        tags: list[str] | None = None,
    ) -> None:
        self.kwargs = {"project": project, "entity": entity, "mode": mode, "tags": tags}
        self.run: Any = None

    def on_fit_start(self, trainer: BaseTrainer) -> None:
        if trainer.is_main_process:
            wandb = require("wandb", extra="tracking")
            self.run = wandb.init(
                config=trainer.run_config,
                dir=trainer.config.output_dir,
                name=Path(trainer.config.output_dir).name,
                **self.kwargs,
            )

    def on_step_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        if self.run:
            self.run.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

    def on_eval_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        if self.run:
            self.run.log(metrics, step=step)

    def on_fit_end(self, trainer: BaseTrainer, metrics: dict[str, float]) -> None:
        if self.run:
            self.run.finish()


@CALLBACKS.register("early_stopping")
class EarlyStopping(Callback):
    """Stops training when an eval metric has not improved for ``patience`` evaluations."""

    def __init__(
        self,
        monitor: str = "eval/loss",
        mode: str = "min",
        patience: int = 3,
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor, self.mode, self.patience, self.min_delta = monitor, mode, patience, min_delta
        self.best: float | None = None
        self.bad_evals = 0

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_eval_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        if self.monitor not in metrics:
            return
        value = metrics[self.monitor]
        if self._improved(value):
            self.best, self.bad_evals = value, 0
        else:
            self.bad_evals += 1
            if self.bad_evals >= self.patience:
                trainer.request_stop(f"{self.monitor} did not improve for {self.patience} evals")


@CALLBACKS.register("throughput")
class Throughput(Callback):
    """Adds ``step_time_s`` and (when batches carry ``input_ids``) ``tokens_per_s``."""

    def __init__(self) -> None:
        self._start = 0.0

    def on_step_start(self, trainer: BaseTrainer, step: int) -> None:
        self._start = time.perf_counter()

    def on_step_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        elapsed = time.perf_counter() - self._start
        metrics["step_time_s"] = elapsed
        if "tokens" in metrics and elapsed > 0:
            metrics["tokens_per_s"] = metrics["tokens"] / elapsed


@CALLBACKS.register("memory_monitor")
class MemoryMonitor(Callback):
    """Adds peak accelerator memory and host RSS (MB) to step metrics."""

    def on_step_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        if torch.cuda.is_available():
            metrics["cuda_max_mem_mb"] = torch.cuda.max_memory_allocated() / BYTES_PER_MB
        metrics["host_max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * RSS_TO_MB


@CALLBACKS.register("torch_profiler")
class TorchProfiler(Callback):
    """Profiles a window of steps with ``torch.profiler`` and exports a Chrome/Perfetto trace
    (open ``<output_dir>/profiler/*.json`` in https://ui.perfetto.dev)."""

    def __init__(
        self,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        record_shapes: bool = True,
        profile_memory: bool = True,
    ) -> None:
        self.schedule = {"wait": wait, "warmup": warmup, "active": active}
        self.record_shapes, self.profile_memory = record_shapes, profile_memory
        self.profiler: Any = None

    def on_fit_start(self, trainer: BaseTrainer) -> None:
        from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        out = Path(trainer.config.output_dir) / "profiler"
        self.profiler = profile(
            activities=activities,
            schedule=schedule(**self.schedule),
            on_trace_ready=tensorboard_trace_handler(str(out)),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
        )
        self.profiler.start()

    def on_step_end(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        if self.profiler:
            self.profiler.step()

    def on_fit_end(self, trainer: BaseTrainer, metrics: dict[str, float]) -> None:
        if self.profiler:
            self.profiler.stop()
            self.profiler = None


@CALLBACKS.register("sample_logger")
class SampleLogger(Callback):
    """Writes a few RL rollouts per step to ``<output_dir>/samples.jsonl`` for inspection."""

    def __init__(self, num_samples: int = 2, every: int = 10) -> None:
        self.num_samples, self.every = num_samples, every

    def on_rollout_end(self, trainer: BaseTrainer, rollout: Any, rewards: torch.Tensor) -> None:
        if not trainer.is_main_process or trainer.step % self.every:
            return
        path = Path(trainer.config.output_dir) / "samples.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for i in range(min(self.num_samples, len(rollout.completions))):
                fh.write(
                    json.dumps(
                        {
                            "step": trainer.step,
                            "prompt": rollout.prompts[i],
                            "completion": rollout.completions[i],
                            "reference": rollout.references[i],
                            "reward": float(rewards[i]),
                        }
                    )
                    + "\n"
                )
