"""The causal-LM interface every language model in the toolbox implements.

Objectives, trainers, rollout and serving code depend on this small interface — never on a
concrete class — so a from-scratch :class:`~toolbox.nn.models.gpt.TinyGPT` and a Hugging Face
model (through :class:`~toolbox.nn.models.hf.HFCausalLM`) are interchangeable
(pattern: *Replace Conditional with Polymorphism*).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class CausalLM(nn.Module, ABC):
    """A model mapping token ids ``(B, T)`` to next-token logits ``(B, T, V)``."""

    max_seq_len: int

    @abstractmethod
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return logits of shape ``(batch, seq_len, vocab_size)``."""

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. Returns only the generated tokens ``(B, max_new_tokens)``.

        Prompts must be **left-padded** to a common length. This reference implementation
        recomputes the full prefix each step (no KV cache) to stay easy to read — production
        inference goes through :mod:`toolbox.serving` (SGLang / vLLM).
        """
        was_training = self.training
        self.eval()
        tokens = input_ids
        finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        fill = pad_token_id if pad_token_id is not None else (eos_token_id or 0)
        generated: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            context = tokens[:, -self.max_seq_len :]
            mask = (context != pad_token_id) if pad_token_id is not None else None
            logits = self(context, attention_mask=mask)[:, -1, :]
            next_token = sample_next_token(logits, temperature, top_k)
            next_token = torch.where(finished, torch.full_like(next_token, fill), next_token)
            generated.append(next_token)
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            if eos_token_id is not None:
                finished |= next_token == eos_token_id
                if bool(finished.all()):
                    break
        self.train(was_training)
        out = torch.stack(generated, dim=1)
        if out.shape[1] < max_new_tokens:  # early stop: pad to a fixed width
            pad = torch.full((out.shape[0], max_new_tokens - out.shape[1]), fill, device=out.device)
            out = torch.cat([out, pad], dim=1)
        return out


def sample_next_token(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None
) -> torch.Tensor:
    """Sample from ``(B, V)`` logits; ``temperature == 0`` means greedy decoding."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
        logits = logits.masked_fill(logits < kth, -torch.inf)
    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
