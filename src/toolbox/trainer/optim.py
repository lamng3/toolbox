"""Optimizer and LR-scheduler factories (pattern: *Factory*)."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from toolbox.core.constants import DEFAULT_ADAM_BETAS, DEFAULT_ADAM_EPS, MIN_LR_RATIO
from toolbox.core.registry import OPTIMIZERS, SCHEDULERS

NO_DECAY_NDIM = 1
"""Parameters with ≤ 1 dimension (biases, norm scales) are excluded from weight decay."""


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= NO_DECAY_NDIM else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@OPTIMIZERS.register("adamw")
def adamw(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Iterable[float] = DEFAULT_ADAM_BETAS,
    eps: float = DEFAULT_ADAM_EPS,
    fused: bool | None = None,
) -> torch.optim.Optimizer:
    """AdamW with weight decay applied only to matrices (not biases or norm scales)."""
    b1, b2 = tuple(betas)
    return torch.optim.AdamW(
        param_groups(model, weight_decay), lr=lr, betas=(b1, b2), eps=eps, fused=fused
    )


@OPTIMIZERS.register("adam")
def adam(model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Adam with L2 weight decay (the classic GNN setting)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


@OPTIMIZERS.register("sgd")
def sgd(
    model: nn.Module, lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    """SGD with momentum."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def _warmup(step: int, warmup_steps: int) -> float:
    return (step + 1) / warmup_steps if step < warmup_steps else 1.0


@SCHEDULERS.register("constant")
def constant(
    optimizer: torch.optim.Optimizer, max_steps: int, warmup_steps: int = 0
) -> LRScheduler:
    """Linear warmup, then a constant learning rate."""
    return LambdaLR(optimizer, lambda step: _warmup(step, warmup_steps))


@SCHEDULERS.register("cosine")
def cosine(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = MIN_LR_RATIO,
) -> LRScheduler:
    """Linear warmup, then cosine decay to ``lr * min_lr_ratio``."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return _warmup(step, warmup_steps)
        progress = min((step - warmup_steps) / max(1, max_steps - warmup_steps), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


@SCHEDULERS.register("linear")
def linear(optimizer: torch.optim.Optimizer, max_steps: int, warmup_steps: int = 0) -> LRScheduler:
    """Linear warmup, then linear decay to zero."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return _warmup(step, warmup_steps)
        return max(0.0, (max_steps - step) / max(1, max_steps - warmup_steps))

    return LambdaLR(optimizer, factor)
