"""Mixed-precision handling, isolated from the training loop."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class Precision:
    """Owns autocast and (for fp16) gradient scaling.

    The training loop calls ``autocast()``, ``backward()`` and ``step()`` without caring which
    precision is active.
    """

    def __init__(self, precision: str, device: torch.device) -> None:
        if precision not in DTYPES:
            raise ValueError(f"precision must be one of {sorted(DTYPES)}, got {precision!r}")
        self.dtype = DTYPES[precision]
        self.device = device
        self.enabled = self.dtype != torch.float32
        use_scaler = self.dtype == torch.float16 and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.enabled):
            yield

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def clip_and_step(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: list[torch.nn.Parameter],
        max_norm: float,
    ) -> float:
        """Unscale, clip, step; returns the pre-clip global gradient norm."""
        self.scaler.unscale_(optimizer)
        if max_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        else:
            grads = [p.grad.norm() for p in parameters if p.grad is not None]
            norm = torch.norm(torch.stack(grads)) if grads else torch.zeros(())
        self.scaler.step(optimizer)
        self.scaler.update()
        return float(norm)
