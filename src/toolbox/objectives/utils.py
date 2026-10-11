"""Log-probability helpers shared by preference, policy and decision objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from toolbox.core.constants import IGNORE_INDEX


def shift_for_next_token(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align position ``t`` logits with token ``t+1`` labels."""
    return logits[:, :-1, :], labels[:, 1:]


def token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Log-prob of each label token, ``(B, T)``; ignored positions get 0.

    Uses ``log_softmax`` + ``gather`` in float32 for numerical stability with bf16 logits.
    """
    safe_labels = labels.clamp(min=0)
    logps = torch.log_softmax(logits.float(), dim=-1)
    out = logps.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return out * (labels != IGNORE_INDEX)


def sequence_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    average: bool = False,
) -> torch.Tensor:
    """Sum (or mean) of log-probs of the *labelled* tokens of each sequence, ``(B,)``."""
    logits, shifted = shift_for_next_token(model(input_ids, attention_mask=attention_mask), labels)
    per_token = token_logprobs(logits, shifted)
    total = per_token.sum(-1)
    if average:
        return total / (shifted != IGNORE_INDEX).sum(-1).clamp(min=1)
    return total


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    mask = mask.to(values.dtype)
    if dim is None:
        return (values * mask).sum() / mask.sum().clamp(min=1)
    return (values * mask).sum(dim) / mask.sum(dim).clamp(min=1)


def cross_entropy_lm(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits, labels = shift_for_next_token(logits, labels)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )
