"""Small shared types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

Batch = dict[str, Any]
"""A mini-batch: a mapping from field name to tensor (or other collated value)."""

Metrics = dict[str, float]


@dataclass
class LossOutput:
    """What every objective returns: a differentiable loss plus scalar metrics to log."""

    loss: torch.Tensor
    metrics: Metrics = field(default_factory=dict)

    def detached_metrics(self, prefix: str = "") -> Metrics:
        out = {f"{prefix}loss": float(self.loss.detach())}
        out.update({f"{prefix}{k}": float(v) for k, v in self.metrics.items()})
        return out


def to_device(batch: Mapping[str, Any], device: torch.device | str) -> Batch:
    """Move every tensor in ``batch`` to ``device`` (non-tensors are passed through)."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
