"""Typed trainer configuration shared by every trainer."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from toolbox.core.constants import DEFAULT_GRAD_CLIP_NORM


@dataclass
class TrainerConfig:
    """Loop-level settings. Trainer-specific settings (``group_size``, ``beta`` ...) are
    separate constructor arguments of each trainer.

    Attributes:
        max_steps: optimizer steps to run.
        batch_size: per-device micro-batch size.
        grad_accum_steps: micro-batches per optimizer step.
        grad_clip: max global grad norm (``0`` disables clipping).
        precision: ``fp32`` | ``bf16`` | ``fp16`` (mixed precision via autocast).
        device: ``auto`` | ``cpu`` | ``cuda`` | ``mps``.
        log_every / eval_every / save_every: step intervals (``0`` disables).
        eval_batches: cap on eval batches (``0`` = full eval set).
        keep_last: number of rolling checkpoints to keep (``last.pt`` is always kept).
        resume_from: checkpoint path, or ``"auto"`` to resume from ``output_dir`` if present.
    """

    max_steps: int = 100
    batch_size: int = 8
    grad_accum_steps: int = 1
    grad_clip: float = DEFAULT_GRAD_CLIP_NORM
    precision: str = "fp32"
    device: str = "auto"
    log_every: int = 10
    eval_every: int = 0
    eval_batches: int = 0
    save_every: int = 0
    keep_last: int = 2
    num_workers: int = 0
    output_dir: str = "runs/default"
    resume_from: str | None = None

    @classmethod
    def split(cls, kwargs: dict[str, Any]) -> tuple[TrainerConfig, dict[str, Any]]:
        """Separate loop-level fields from trainer-specific kwargs."""
        names = {f.name for f in dataclasses.fields(cls)}
        common = {k: v for k, v in kwargs.items() if k in names}
        extra = {k: v for k, v in kwargs.items() if k not in names}
        return cls(**common), extra
