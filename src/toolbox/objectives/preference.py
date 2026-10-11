"""Preference-optimisation losses (offline RLHF) and the Bradley–Terry reward-model loss.

All DPO-family losses are functions of the *implicit reward margin*

    h = β · [(log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))]

and differ only in the function applied to ``h``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from toolbox.core.registry import OBJECTIVES
from toolbox.core.types import Batch, LossOutput
from toolbox.objectives.base import PreferenceObjective, SupervisedObjective


class _MarginLoss(PreferenceObjective):
    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def _loss_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self,
        policy_chosen: torch.Tensor,
        policy_rejected: torch.Tensor,
        ref_chosen: torch.Tensor,
        ref_rejected: torch.Tensor,
    ) -> LossOutput:
        chosen_reward = policy_chosen - ref_chosen
        rejected_reward = policy_rejected - ref_rejected
        logits = chosen_reward - rejected_reward
        loss = self._loss_from_logits(logits).mean()
        logits, chosen_reward, rejected_reward = (
            t.detach() for t in (logits, chosen_reward, rejected_reward)
        )
        return LossOutput(
            loss,
            {
                "reward_margin": float((self.beta * logits).mean()),
                "reward_accuracy": float((logits > 0).float().mean()),
                "chosen_reward": float((self.beta * chosen_reward).mean()),
                "rejected_reward": float((self.beta * rejected_reward).mean()),
            },
        )


@OBJECTIVES.register("dpo")
class DPOLoss(_MarginLoss):
    """Direct Preference Optimization (Rafailov et al., 2023), with conservative label smoothing.

    ``label_smoothing = ε`` assumes preference labels are flipped with probability ε (cDPO).
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0) -> None:
        super().__init__(beta)
        self.label_smoothing = label_smoothing

    def _loss_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        h = self.beta * logits
        eps = self.label_smoothing
        return -(1 - eps) * F.logsigmoid(h) - eps * F.logsigmoid(-h)


@OBJECTIVES.register("ipo")
class IPOLoss(_MarginLoss):
    """Identity Preference Optimisation (Azar et al., 2023): regress the margin to ``1/(2β)``."""

    def _loss_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits - 1 / (2 * self.beta)) ** 2


@OBJECTIVES.register("slic")
class HingeLoss(_MarginLoss):
    """SLiC-HF style hinge loss on the calibrated margin (Zhao et al., 2023)."""

    def _loss_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.relu(1 - self.beta * logits)


@OBJECTIVES.register("reward_bt")
class BradleyTerryLoss(SupervisedObjective):
    """Train a scalar reward model: ``-log σ(r(x, y_w) − r(x, y_l))``."""

    def __call__(self, model: nn.Module, batch: Batch) -> LossOutput:
        chosen = model(batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"])
        rejected = model(
            batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]
        )
        margin = chosen - rejected
        loss = -F.logsigmoid(margin).mean()
        margin = margin.detach()
        return LossOutput(
            loss,
            {
                "accuracy": float((margin > 0).float().mean()),
                "margin": float(margin.mean()),
            },
        )
