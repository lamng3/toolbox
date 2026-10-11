"""Decoder-only transformer building blocks (Llama-style).

Readable, dependency-free implementations meant to be studied and modified: RMSNorm, rotary
position embeddings, grouped-query causal self-attention (using PyTorch's fused
``scaled_dot_product_attention``), SwiGLU feed-forward and a pre-norm residual block.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from toolbox.core.constants import NORM_EPS, ROPE_BASE
from toolbox.core.registry import ATTENTIONS, FEEDFORWARDS, NORMS
from toolbox.nn.layers.common import LayerSpec, build_layer


@NORMS.register("rmsnorm")
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019): no mean-centering, no bias."""

    def __init__(self, dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE). Caches cos/sin tables up to ``max_seq_len``."""

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, head_dim: int, max_seq_len: int, base: float = ROPE_BASE) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # (T, D/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Rotate ``x`` of shape ``(B, H, T, D)``."""
        seq_len = x.shape[-2]
        cos = self.cos[offset : offset + seq_len].to(x.dtype)
        sin = self.sin[offset : offset + seq_len].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rotated.flatten(-2)


@ATTENTIONS.register("causal_self_attention")
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional grouped-query attention (GQA)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        if n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(batch, seq_len, 2, self.n_kv_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        mask = None
        if attention_mask is not None:
            # Combine causal mask with key padding mask -> (B, 1, T, T) boolean "may attend".
            # The diagonal is always allowed: a fully-masked row (a padding query) would
            # otherwise produce NaNs that leak into real tokens through 0 * NaN in later layers.
            causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
            diagonal = torch.eye(seq_len, dtype=torch.bool, device=x.device)
            keys = attention_mask[:, None, None, :].bool()
            mask = causal[None, None] & (keys | diagonal[None, None])
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(out)


@FEEDFORWARDS.register("swiglu")
class SwiGLU(nn.Module):
    """Gated feed-forward: ``W2(silu(W1 x) * W3 x)`` (Shazeer, 2020)."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


@FEEDFORWARDS.register("gelu_mlp")
class GeluMLP(nn.Module):
    """Classic transformer feed-forward: ``W2(gelu(W1 x))`` (GPT-2 / BERT)."""

    def __init__(self, d_model: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm residual block: ``x + attn(norm(x))`` then ``x + ffn(norm(x))``.

    ``attention``, ``ffn`` and ``norm`` are layer specs (registered names or
    ``{"name": ..., **kwargs}``), so any registered implementation can replace the defaults.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        ffn_hidden_dim: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        attention: LayerSpec = "causal_self_attention",
        ffn: LayerSpec = "swiglu",
        norm: LayerSpec = "rmsnorm",
    ) -> None:
        super().__init__()
        self.attn_norm = build_layer(NORMS, norm, dim=d_model)
        self.attn = build_layer(
            ATTENTIONS,
            attention,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
        )
        self.ffn_norm = build_layer(NORMS, norm, dim=d_model)
        self.ffn = build_layer(FEEDFORWARDS, ffn, d_model=d_model, hidden_dim=ffn_hidden_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask)
        return x + self.ffn(self.ffn_norm(x))
