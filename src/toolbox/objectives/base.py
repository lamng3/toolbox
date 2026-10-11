"""Objective interfaces (pattern: *Strategy*).

Each training paradigm needs a different *shape* of loss, so there are three families. A
trainer depends on one family's interface and any registered implementation can be swapped in
from config (``objective.name=ipo``) — no ``if loss_type == ...`` chains in the trainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from toolbox.core.types import Batch, LossOutput


class SupervisedObjective(ABC):
    """``loss = f(model, batch)`` — LM pretraining, SFT, reward modelling, GNN training."""

    @abstractmethod
    def __call__(self, model: nn.Module, batch: Batch) -> LossOutput: ...


class PreferenceObjective(ABC):
    """Pairwise loss on sequence log-probs of chosen/rejected answers (DPO family)."""

    requires_reference: bool = True

    @abstractmethod
    def __call__(
        self,
        policy_chosen: torch.Tensor,
        policy_rejected: torch.Tensor,
        ref_chosen: torch.Tensor,
        ref_rejected: torch.Tensor,
    ) -> LossOutput: ...


class PolicyObjective(ABC):
    """Policy-gradient loss on per-token log-probs and advantages (PPO / GRPO family).

    All tensors are ``(B, T)`` except ``advantages`` which may be ``(B,)`` (sequence-level,
    broadcast over tokens) or ``(B, T)``.
    """

    @abstractmethod
    def __call__(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        ref_logprobs: torch.Tensor | None = None,
    ) -> LossOutput: ...
