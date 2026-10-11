"""A small Llama-style GPT for from-scratch experiments, plus a reward model.

The block internals are layer specs: ``attention`` (``ATTENTIONS``), ``ffn`` (``FEEDFORWARDS``)
and ``norm`` (``NORMS``). Swap them to test architectures, e.g.
``model.ffn=gelu_mlp model.norm=layernorm``, or register your own and select it by name.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from toolbox.core.registry import MODELS, NORMS
from toolbox.nn.layers import LayerSpec, ScalarHead, TransformerBlock, build_layer
from toolbox.nn.models.base import CausalLM

FFN_ROUND_TO = 32
"""SwiGLU hidden size is rounded up to a multiple of this for hardware-friendly matmuls."""

INIT_STD = 0.02


GATED_FFNS = frozenset({"swiglu"})


def swiglu_hidden_dim(d_model: int, ffn_mult: float) -> int:
    # 2/3 factor keeps the parameter count of a gated FFN close to an ungated 4x MLP.
    hidden = int(2 * ffn_mult * d_model / 3)
    return FFN_ROUND_TO * math.ceil(hidden / FFN_ROUND_TO)


def ffn_hidden_dim(d_model: int, ffn_mult: float, ffn: LayerSpec) -> int:
    """Hidden width for an FFN spec: gated FFNs use the 2/3 rule, others ``ffn_mult * d``."""
    name = ffn if isinstance(ffn, str) else str(ffn["name"])
    if name in GATED_FFNS:
        return swiglu_hidden_dim(d_model, ffn_mult)
    return FFN_ROUND_TO * math.ceil(int(ffn_mult * d_model) / FFN_ROUND_TO)


@MODELS.register("tiny_gpt")
class TinyGPT(CausalLM):
    """Decoder-only transformer: token embedding → N pre-norm blocks → norm → LM head.

    Args:
        attention: attention spec for every block (default: causal attention with RoPE).
        ffn: feed-forward spec for every block (default: SwiGLU).
        norm: norm spec for every block and the final norm (default: RMSNorm).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_kv_heads: int | None = None,
        max_seq_len: int = 256,
        ffn_mult: float = 4.0,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        attention: LayerSpec = "causal_self_attention",
        ffn: LayerSpec = "swiglu",
        norm: LayerSpec = "rmsnorm",
    ) -> None:
        super().__init__()
        self.vocab_size, self.d_model, self.max_seq_len = vocab_size, d_model, max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        hidden = ffn_hidden_dim(d_model, ffn_mult, ffn)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model,
                n_heads,
                max_seq_len,
                hidden,
                n_kv_heads,
                dropout,
                attention=attention,
                ffn=ffn,
                norm=norm,
            )
            for _ in range(n_layers)
        )
        self.norm = build_layer(NORMS, norm, dim=d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.norm(x)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.lm_head(self.hidden_states(input_ids, attention_mask))

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


@MODELS.register("tiny_reward_model")
class SequenceRewardModel(nn.Module):
    """A transformer backbone with a scalar head read at the last non-padding token.

    Trained with the Bradley–Terry objective (``reward_bt``) on preference pairs; used as a
    learned reward in RLHF via :class:`toolbox.rewards.ModelReward`.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 256,
        attention: LayerSpec = "causal_self_attention",
        ffn: LayerSpec = "swiglu",
        norm: LayerSpec = "rmsnorm",
        head: LayerSpec = "linear",
    ) -> None:
        super().__init__()
        self.backbone = TinyGPT(
            vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            attention=attention,
            ffn=ffn,
            norm=norm,
        )
        self.head = ScalarHead(d_model, head)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.backbone.hidden_states(input_ids, attention_mask)
        scores = self.head(hidden)  # (B, T)
        if attention_mask is None:
            return scores[:, -1]
        last = attention_mask.long().sum(dim=1).clamp(min=1) - 1  # right-padded inputs
        return scores.gather(1, last[:, None]).squeeze(1)
