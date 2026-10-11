"""Generic layers (activations, norms, MLPs), registered so models can swap them by name.

Every model slot that holds one of these takes a *spec*: a registered name (``"gelu"``) or a
mapping (``{"name": "mlp", "hidden_dims": [64]}``). Register your own with the matching
decorator, e.g. ``@MLPS.register("my_mlp")``, and select it from config.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from toolbox.core.registry import ACTIVATIONS, MLPS, NORMS, Registry

LayerSpec = str | Mapping[str, Any]
"""A registered layer name, or ``{"name": ..., **kwargs}``."""

NORM_EPS_DEFAULT = 1e-5


def build_layer(registry: Registry[Any], spec: LayerSpec | None, **required: Any) -> nn.Module:
    """Build ``spec`` from ``registry``.

    ``required`` are the role's contract arguments (``in_dim``/``out_dim``, ``dim`` ...) and
    always win over values in the spec. ``None`` builds an identity, so optional slots such as
    ``norm: null`` need no special casing in models.
    """
    if spec is None:
        return nn.Identity()
    module: nn.Module = registry.build(spec, **required)
    return module


# -- activations: (**kw) → f(x) ------------------------------------------------------------
ACTIVATIONS.add("relu", nn.ReLU)
ACTIVATIONS.add("gelu", nn.GELU)
ACTIVATIONS.add("silu", nn.SiLU)
ACTIVATIONS.add("tanh", nn.Tanh)
ACTIVATIONS.add("identity", nn.Identity)


# -- norms: (dim, **kw) → f(x) -------------------------------------------------------------
@NORMS.register("layernorm")
def layernorm(dim: int, eps: float = NORM_EPS_DEFAULT) -> nn.Module:
    """LayerNorm over the last dimension."""
    return nn.LayerNorm(dim, eps=eps)


@NORMS.register("batchnorm")
def batchnorm(dim: int) -> nn.Module:
    """BatchNorm over the feature dimension of ``(N, dim)`` inputs (e.g. node features)."""
    return nn.BatchNorm1d(dim)


@NORMS.register("identity")
def identity_norm(dim: int) -> nn.Module:
    """No normalisation."""
    return nn.Identity()


# -- MLPs: (in_dim, out_dim, **kw) → f(x) --------------------------------------------------
@MLPS.register("linear")
def linear(in_dim: int, out_dim: int, bias: bool = True) -> nn.Module:
    """A single linear projection."""
    return nn.Linear(in_dim, out_dim, bias=bias)


@MLPS.register("mlp")
class MLP(nn.Module):
    """``Linear → [norm] → activation → [dropout]`` per hidden layer; the last layer is linear."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (64,),
        activation: LayerSpec = "relu",
        norm: LayerSpec | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            if norm is not None:
                layers.append(build_layer(NORMS, norm, dim=hidden))
            layers.append(build_layer(ACTIVATIONS, activation))
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarHead(nn.Module):
    """Projects hidden states to one scalar per position (value heads, reward heads)."""

    def __init__(self, d_model: int, head: LayerSpec = "linear") -> None:
        super().__init__()
        self.proj = build_layer(MLPS, head, in_dim=d_model, out_dim=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).squeeze(-1)
