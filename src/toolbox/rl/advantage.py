"""Advantage estimators.

Sequence-level estimators for LLM RL are registered strategies (``ADVANTAGES``); they turn the
rewards of ``G`` sampled completions per prompt into per-completion advantages. GAE for
environment RL needs values and episode boundaries, so it is a plain function.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from toolbox.core.constants import DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA, EPS
from toolbox.core.registry import ADVANTAGES


class AdvantageEstimator(ABC):
    """Map rewards ``(num_prompts * group_size,)`` (grouped contiguously) to advantages."""

    @abstractmethod
    def __call__(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor: ...


@ADVANTAGES.register("group_relative")
class GroupRelativeAdvantage(AdvantageEstimator):
    """GRPO: ``(r − mean_group) / (std_group + ε)``. ``scale=False`` gives Dr. GRPO's unscaled
    variant, which avoids up-weighting near-deterministic groups."""

    def __init__(self, scale: bool = True) -> None:
        self.scale = scale

    def __call__(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor:
        grouped = rewards.view(-1, group_size)
        centered = grouped - grouped.mean(dim=1, keepdim=True)
        if self.scale:
            centered = centered / (grouped.std(dim=1, keepdim=True) + EPS)
        return centered.flatten()


@ADVANTAGES.register("rloo")
class LeaveOneOutAdvantage(AdvantageEstimator):
    """RLOO (Ahmadian et al., 2024): baseline is the mean reward of the *other* samples."""

    def __call__(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor:
        if group_size < 2:
            raise ValueError("RLOO needs group_size >= 2")
        grouped = rewards.view(-1, group_size)
        baseline = (grouped.sum(dim=1, keepdim=True) - grouped) / (group_size - 1)
        return (grouped - baseline).flatten()


@ADVANTAGES.register("batch_normalized")
class BatchNormalizedAdvantage(AdvantageEstimator):
    """REINFORCE++-style: whiten rewards across the whole batch, ignoring groups."""

    def __call__(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor:
        return (rewards - rewards.mean()) / (rewards.std() + EPS)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = DEFAULT_GAMMA,
    lam: float = DEFAULT_GAE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised Advantage Estimation (Schulman et al., 2016) over a ``(T,)`` trajectory.

    ``dones[t] = 1`` means the episode ended *after* step ``t``. Returns
    ``(advantages, returns)`` with ``returns = advantages + values``.
    """
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(())
    next_value = last_value
    for t in reversed(range(rewards.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        next_advantage = delta + gamma * lam * not_done * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages, advantages + values
