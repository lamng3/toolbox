"""Adapter exposing a Hugging Face ``AutoModelForCausalLM`` through :class:`CausalLM`.

Pattern: *Adapter*. The rest of the toolbox only speaks ``model(input_ids, attention_mask) ->
logits``; this class translates to and from the ``transformers`` API (``.logits``, ``generate``
keyword names, LoRA via ``peft``).
"""

from __future__ import annotations

from typing import Any

import torch

from toolbox.core.decorators import require
from toolbox.core.registry import MODELS
from toolbox.nn.models.base import CausalLM

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@MODELS.register("hf_causal_lm")
class HFCausalLM(CausalLM):
    """Wrap any Hub/local causal LM, optionally with LoRA adapters.

    Args:
        model_name_or_path: e.g. ``"Qwen/Qwen2.5-0.5B-Instruct"``.
        dtype: ``float32`` | ``bfloat16`` | ``float16``.
        lora: optional ``peft.LoraConfig`` kwargs, e.g. ``{"r": 16, "lora_alpha": 32}``.
        gradient_checkpointing: trade compute for memory on long sequences.
        vocab_size: ignored (accepted so the builder can inject it uniformly).
    """

    def __init__(
        self,
        model_name_or_path: str,
        dtype: str = "bfloat16",
        lora: dict[str, Any] | None = None,
        gradient_checkpointing: bool = False,
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        transformers = require("transformers", extra="llm")
        kwargs: dict[str, Any] = {
            "torch_dtype": DTYPES[dtype],
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if lora:
            peft = require("peft", extra="llm")
            self.model = peft.get_peft_model(self.model, peft.LoraConfig(**lora))
        self.max_seq_len = getattr(self.model.config, "max_position_embeddings", 4096)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

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
        mask = (input_ids != pad_token_id).long() if pad_token_id is not None else None
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_k=top_k,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        new_tokens = out[:, input_ids.shape[1] :]
        if new_tokens.shape[1] < max_new_tokens:
            fill = pad_token_id if pad_token_id is not None else 0
            pad = torch.full(
                (new_tokens.shape[0], max_new_tokens - new_tokens.shape[1]),
                fill,
                device=new_tokens.device,
                dtype=new_tokens.dtype,
            )
            new_tokens = torch.cat([new_tokens, pad], dim=1)
        return new_tokens

    def save_pretrained(self, path: str) -> None:
        self.model.save_pretrained(path)
