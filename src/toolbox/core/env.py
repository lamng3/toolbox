"""Device selection, seeding and environment reporting."""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np
import torch

from toolbox.core.constants import DEFAULT_SEED, ENV_SLURM_JOB_ID

AUTO_DEVICE = "auto"


def resolve_device(device: str = AUTO_DEVICE) -> torch.device:
    """Map ``"auto"`` to the best available accelerator (CUDA → MPS → CPU)."""
    if device != AUTO_DEVICE:
        return torch.device(device)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device("cuda", local_rank)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def collect_env() -> dict[str, Any]:
    """A snapshot of the software/hardware environment — printed by ``toolbox env``."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "slurm_job_id": os.environ.get(ENV_SLURM_JOB_ID),
    }
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    for optional in (
        "transformers",
        "trl",
        "peft",
        "torch_geometric",
        "anyjev",
        "sglang",
        "vllm",
        "wandb",
    ):
        try:
            module = __import__(optional)
            info[optional] = getattr(module, "__version__", "installed")
        except ImportError:
            info[optional] = None
    return info
