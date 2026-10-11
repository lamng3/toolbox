"""Distribution strategies (pattern: *Strategy*).

The trainer asks its strategy to place the model, build data loaders and reduce metrics. Going
from one GPU to many is a config change (``strategy=ddp``) plus a ``torchrun``/SLURM launch —
the training loop is untouched.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from toolbox.core.constants import ENV_WORLD_SIZE
from toolbox.core.env import resolve_device
from toolbox.core.registry import STRATEGIES

AUTO = "auto"


class Strategy(ABC):
    """How the model and data are distributed across processes."""

    def __init__(self, device: str = AUTO) -> None:
        self.device = resolve_device(device)

    def setup(self) -> None:
        """Initialise process groups etc. Called once before training."""

    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module: ...

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return getattr(model, "module", model)

    def dataloader(
        self,
        dataset: Dataset[Any],
        batch_size: int,
        shuffle: bool,
        collate_fn: Any,
        num_workers: int = 0,
    ) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    def reduce_mean(self, value: float) -> float:
        return value

    @property
    def is_main_process(self) -> bool:
        return True

    @property
    def world_size(self) -> int:
        return 1

    def barrier(self) -> None:
        """Synchronise processes (no-op on a single device)."""

    def teardown(self) -> None:
        """Release process groups."""


@STRATEGIES.register("single")
class SingleDevice(Strategy):
    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.to(self.device)


@STRATEGIES.register("ddp")
class DistributedDataParallel(Strategy):
    """Data parallelism with ``torch.nn.parallel.DistributedDataParallel``.

    Expects the standard ``torchrun`` environment (``RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``).
    """

    def __init__(
        self, device: str = AUTO, backend: str | None = None, find_unused_parameters: bool = False
    ) -> None:
        super().__init__(device)
        self.backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        self.find_unused = find_unused_parameters

    def setup(self) -> None:
        if not dist.is_initialized():
            dist.init_process_group(self.backend)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        device_ids = [self.device.index] if self.device.type == "cuda" else None
        return nn.parallel.DistributedDataParallel(
            model, device_ids=device_ids, find_unused_parameters=self.find_unused
        )

    def dataloader(
        self,
        dataset: Dataset[Any],
        batch_size: int,
        shuffle: bool,
        collate_fn: Any,
        num_workers: int = 0,
    ) -> DataLoader[Any]:
        sampler = DistributedSampler(dataset, shuffle=shuffle)  # type: ignore[var-annotated]
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    def reduce_mean(self, value: float) -> float:
        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor) / self.world_size

    @property
    def is_main_process(self) -> bool:
        return dist.get_rank() == 0 if dist.is_initialized() else True

    @property
    def world_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


@STRATEGIES.register("fsdp")
class FullyShardedDataParallel(DistributedDataParallel):
    """Shard parameters, gradients and optimizer state (FSDP) for models that don't fit on one
    GPU. Wraps each transformer block as its own FSDP unit when ``auto_wrap`` is on."""

    def __init__(
        self,
        device: str = AUTO,
        backend: str | None = None,
        mixed_precision: str | None = None,
        auto_wrap: bool = True,
    ) -> None:
        super().__init__(device, backend)
        self.mixed_precision = mixed_precision
        self.auto_wrap = auto_wrap

    def wrap_model(self, model: nn.Module) -> nn.Module:
        import functools

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        from toolbox.nn.layers import TransformerBlock

        policy = (
            functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock}
            )
            if self.auto_wrap
            else None
        )
        mp = None
        if self.mixed_precision == "bf16":
            mp = MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16
            )
        device_id = self.device if self.device.type == "cuda" else None
        return FSDP(
            model.to(self.device), auto_wrap_policy=policy, mixed_precision=mp, device_id=device_id
        )


@STRATEGIES.register(AUTO)
def auto(device: str = AUTO) -> Strategy:
    """DDP when launched with ``WORLD_SIZE > 1`` (torchrun / srun), otherwise single device."""
    if int(os.environ.get(ENV_WORLD_SIZE, 1)) > 1:
        return DistributedDataParallel(device)
    return SingleDevice(device)


__all__ = ["DistributedDataParallel", "FullyShardedDataParallel", "SingleDevice", "Strategy"]
