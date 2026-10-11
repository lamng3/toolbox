"""Text rollouts: sample ``G`` completions per prompt from the current policy.

The rollout returns token ids plus the masks needed to compute per-token log-probs of the
completion only. Generation goes through the model's own ``generate`` (in-process). For large
models, generation is the bottleneck: :mod:`toolbox.serving` adapters (SGLang/vLLM) are the
place to plug a dedicated inference engine with weight syncing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from toolbox.core.constants import IGNORE_INDEX
from toolbox.data.collators import pad_sequences
from toolbox.data.tokenization import Tokenizer
from toolbox.nn.models.base import CausalLM
from toolbox.objectives.utils import shift_for_next_token, token_logprobs


@dataclass
class RolloutBatch:
    """``B = num_prompts * group_size`` sequences of ``prompt ⧺ completion``."""

    input_ids: torch.Tensor  # (B, P + C), prompt left-padded
    attention_mask: torch.Tensor  # (B, P + C)
    labels: torch.Tensor  # (B, P + C); IGNORE_INDEX outside the completion
    prompts: list[str]
    completions: list[str]
    references: list[str]
    group_size: int

    @property
    def completion_mask(self) -> torch.Tensor:
        """``(B, P + C - 1)`` mask aligned with next-token log-probs."""
        return (self.labels[:, 1:] != IGNORE_INDEX).float()


@dataclass
class SamplingConfig:
    group_size: int = 4
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_k: int | None = None


class TextRollout:
    """Generates grouped completions and packs them for policy-gradient training."""

    def __init__(self, tokenizer: Tokenizer, sampling: SamplingConfig) -> None:
        self.tokenizer = tokenizer
        self.sampling = sampling

    @torch.no_grad()
    def __call__(
        self,
        policy: CausalLM,
        prompts: Sequence[str],
        references: Sequence[str],
        device: torch.device,
    ) -> RolloutBatch:
        tok, cfg = self.tokenizer, self.sampling
        group_prompts = [p for p in prompts for _ in range(cfg.group_size)]
        group_refs = [r for r in references for _ in range(cfg.group_size)]
        prompt_ids = pad_sequences(
            [tok.encode(p, add_bos=True) for p in group_prompts], tok.pad_token_id, side="left"
        ).to(device)
        completion_ids = policy.generate(
            prompt_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        completion_mask = self._completion_mask(completion_ids)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_ids != tok.pad_token_id, completion_mask], dim=1).long()
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[:, prompt_ids.shape[1] :] = torch.where(
            completion_mask, completion_ids, torch.full_like(completion_ids, IGNORE_INDEX)
        )
        completions = [
            tok.decode(ids[m].tolist()) for ids, m in zip(completion_ids, completion_mask)
        ]
        return RolloutBatch(
            input_ids,
            attention_mask,
            labels,
            group_prompts,
            completions,
            group_refs,
            cfg.group_size,
        )

    def _completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        """True up to and including the first EOS; False for padding after it."""
        eos = self.tokenizer.eos_token_id
        if eos is None:
            return completion_ids != self.tokenizer.pad_token_id
        is_eos = (completion_ids == eos).long()
        after_eos = (is_eos.cumsum(dim=1) - is_eos) > 0
        return ~after_eos


def rollout_logprobs(model: torch.nn.Module, batch: RolloutBatch) -> torch.Tensor:
    """Per-token log-probs ``(B, T-1)`` of the completion tokens under ``model``."""
    logits = model(batch.input_ids, attention_mask=batch.attention_mask)
    logits, labels = shift_for_next_token(logits, batch.labels)
    return token_logprobs(logits, labels)
