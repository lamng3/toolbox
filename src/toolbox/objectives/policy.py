"""Policy-gradient losses for online RL (PPO, GRPO, REINFORCE).

GRPO and PPO share the clipped surrogate objective; they differ in how advantages are computed
(group-relative vs. GAE with a critic — see :mod:`toolbox.rl.advantage`) and in their default
KL regularisation. That difference lives in configuration, not in duplicated code.
"""

from __future__ import annotations

import torch

from toolbox.core.constants import DEFAULT_CLIP_EPS
from toolbox.core.registry import OBJECTIVES
from toolbox.core.types import LossOutput
from toolbox.objectives.base import PolicyObjective
from toolbox.objectives.utils import masked_mean

TOKEN_MEAN = "token_mean"
SEQ_MEAN_TOKEN_MEAN = "seq_mean_token_mean"


def kl_k3(logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """Unbiased, non-negative KL(π‖π_ref) estimator ``exp(r) − r − 1`` with ``r = ref − π``.

    See J. Schulman, "Approximating KL Divergence" (2020).
    """
    log_ratio = ref_logprobs - logprobs
    return torch.exp(log_ratio) - log_ratio - 1


def aggregate(values: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce ``(B, T)`` token values to a scalar.

    ``token_mean`` weights every token equally (DAPO / Dr. GRPO); ``seq_mean_token_mean``
    averages within each sequence first (original GRPO), which up-weights short answers.
    """
    if mode == TOKEN_MEAN:
        return masked_mean(values, mask)
    if mode == SEQ_MEAN_TOKEN_MEAN:
        return masked_mean(values, mask, dim=-1).mean()
    raise ValueError(f"Unknown loss aggregation '{mode}'")


@OBJECTIVES.register("ppo_clip")
class ClippedPolicyLoss(PolicyObjective):
    """PPO clipped surrogate (Schulman et al., 2017) with optional KL penalty to a reference.

    Args:
        clip_eps: lower clip range; ``clip_eps_high`` defaults to the same (set higher for
            DAPO-style "clip-higher").
        kl_coef: weight of the k3 KL penalty; 0 disables it and the reference model.
        loss_agg: ``token_mean`` or ``seq_mean_token_mean``.
    """

    def __init__(
        self,
        clip_eps: float = DEFAULT_CLIP_EPS,
        clip_eps_high: float | None = None,
        kl_coef: float = 0.0,
        loss_agg: str = TOKEN_MEAN,
    ) -> None:
        self.clip_low = clip_eps
        self.clip_high = clip_eps if clip_eps_high is None else clip_eps_high
        self.kl_coef = kl_coef
        self.loss_agg = loss_agg

    @property
    def requires_reference(self) -> bool:
        return self.kl_coef > 0

    def __call__(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        ref_logprobs: torch.Tensor | None = None,
    ) -> LossOutput:
        if advantages.dim() == 1:
            advantages = advantages[:, None].expand_as(logprobs)
        ratio = torch.exp(logprobs - old_logprobs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - self.clip_low, 1 + self.clip_high) * advantages
        per_token = -torch.minimum(unclipped, clipped)
        metrics = {
            "clip_fraction": float(masked_mean((unclipped != clipped).float(), mask)),
            "ratio_mean": float(masked_mean(ratio.detach(), mask)),
        }
        if self.kl_coef > 0:
            if ref_logprobs is None:
                raise ValueError("kl_coef > 0 requires ref_logprobs")
            kl = kl_k3(logprobs, ref_logprobs)
            per_token = per_token + self.kl_coef * kl
            metrics["kl"] = float(masked_mean(kl.detach(), mask))
        return LossOutput(aggregate(per_token, mask, self.loss_agg), metrics)


@OBJECTIVES.register("grpo")
class GRPOLoss(ClippedPolicyLoss):
    """Group Relative Policy Optimization (Shao et al., 2024) defaults: KL 0.04, per-seq mean."""

    def __init__(
        self,
        clip_eps: float = DEFAULT_CLIP_EPS,
        clip_eps_high: float | None = None,
        kl_coef: float = 0.04,
        loss_agg: str = SEQ_MEAN_TOKEN_MEAN,
    ) -> None:
        super().__init__(clip_eps, clip_eps_high, kl_coef, loss_agg)


@OBJECTIVES.register("reinforce")
class ReinforceLoss(PolicyObjective):
    """Vanilla policy gradient ``-A · log π`` (use with RLOO / baseline advantages)."""

    def __init__(self, loss_agg: str = TOKEN_MEAN) -> None:
        self.loss_agg = loss_agg

    requires_reference = False

    def __call__(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        ref_logprobs: torch.Tensor | None = None,
    ) -> LossOutput:
        if advantages.dim() == 1:
            advantages = advantages[:, None].expand_as(logprobs)
        return LossOutput(aggregate(-advantages * logprobs, mask, self.loss_agg))
