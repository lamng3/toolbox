"""Reusable neural-network layers. Each is registered under its role so models can swap it.

| Role (registry) | Built-ins |
| --- | --- |
| ``graph_conv`` | ``gcn``, ``sage``, ``gat`` |
| ``mlp`` | ``linear``, ``mlp`` |
| ``norm`` | ``rmsnorm``, ``layernorm``, ``batchnorm``, ``identity`` |
| ``activation`` | ``relu``, ``gelu``, ``silu``, ``tanh``, ``identity`` |
| ``feedforward`` | ``swiglu``, ``gelu_mlp`` |
| ``attention`` | ``causal_self_attention`` |
"""

from toolbox.nn.layers.common import MLP, LayerSpec, ScalarHead, build_layer
from toolbox.nn.layers.graph import GATConv, GCNConv, SAGEConv
from toolbox.nn.layers.transformer import (
    CausalSelfAttention,
    GeluMLP,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
)

__all__ = [
    "MLP",
    "CausalSelfAttention",
    "GATConv",
    "GCNConv",
    "GeluMLP",
    "LayerSpec",
    "RMSNorm",
    "RotaryEmbedding",
    "SAGEConv",
    "ScalarHead",
    "SwiGLU",
    "TransformerBlock",
    "build_layer",
]
