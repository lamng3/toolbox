"""Named constants used across the toolbox.

Every "magic number" that would otherwise be scattered through the code base lives here
(pattern: *Remove Magic Numbers*). Values that users are expected to tune belong in the
Hydra configs instead; this module only holds conventions and safe library defaults.
"""

from __future__ import annotations

from typing import Final

# --- Tokenisation / loss conventions -------------------------------------------------------
IGNORE_INDEX: Final[int] = -100
"""Label value ignored by ``torch.nn.functional.cross_entropy`` (PyTorch / HF convention)."""

BYTE_VOCAB_SIZE: Final[int] = 256
"""Number of raw byte values handled by :class:`toolbox.data.tokenization.ByteTokenizer`."""

# --- Reproducibility -----------------------------------------------------------------------
DEFAULT_SEED: Final[int] = 42

# --- Optimisation --------------------------------------------------------------------------
DEFAULT_GRAD_CLIP_NORM: Final[float] = 1.0
DEFAULT_ADAM_BETAS: Final[tuple[float, float]] = (0.9, 0.95)
DEFAULT_ADAM_EPS: Final[float] = 1e-8
MIN_LR_RATIO: Final[float] = 0.1
"""Cosine schedules decay to ``lr * MIN_LR_RATIO`` rather than to zero."""

# --- Numerics ------------------------------------------------------------------------------
EPS: Final[float] = 1e-8
"""Small constant added to denominators (advantage normalisation, cosine similarity, ...)."""

NORM_EPS: Final[float] = 1e-6
"""Epsilon used inside normalisation layers (RMSNorm / LayerNorm)."""

ROPE_BASE: Final[float] = 10_000.0
"""Base frequency of rotary position embeddings (Su et al., 2021)."""

# --- RL ------------------------------------------------------------------------------------
DEFAULT_GAMMA: Final[float] = 0.99
DEFAULT_GAE_LAMBDA: Final[float] = 0.95
DEFAULT_CLIP_EPS: Final[float] = 0.2

# --- Checkpointing & logging ---------------------------------------------------------------
CHECKPOINT_DIRNAME: Final[str] = "checkpoints"
LAST_CHECKPOINT_NAME: Final[str] = "last.pt"
RESOLVED_CONFIG_NAME: Final[str] = "config.yaml"
METRICS_FILENAME: Final[str] = "metrics.jsonl"

# --- Plugins -------------------------------------------------------------------------------
PLUGIN_ENTRY_POINT_GROUP: Final[str] = "toolbox.plugins"

# --- Distributed environment variables (set by torchrun / SLURM) ---------------------------
ENV_WORLD_SIZE: Final[str] = "WORLD_SIZE"
ENV_RANK: Final[str] = "RANK"
ENV_LOCAL_RANK: Final[str] = "LOCAL_RANK"
ENV_SLURM_JOB_ID: Final[str] = "SLURM_JOB_ID"
