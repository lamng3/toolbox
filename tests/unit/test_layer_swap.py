"""Every model slot accepts any registered layer, including ones defined outside toolbox."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from toolbox.core.registry import (
    ACTIVATIONS,
    ATTENTIONS,
    FEEDFORWARDS,
    GRAPH_CONVS,
    MLPS,
    MODELS,
    NORMS,
    RegistryError,
)
from toolbox.experiment import ExperimentBuilder
from toolbox.nn.layers import build_layer
from toolbox.nn.models import ActorCritic, GNNNodeClassifier, TinyGPT

CALLS: dict[str, int] = {}


def _count(name: str) -> None:
    CALLS[name] = CALLS.get(name, 0) + 1


@MLPS.register("test_counting_mlp", override=True)
class CountingMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale: float = 1.0) -> None:
        super().__init__()
        self.proj, self.scale = nn.Linear(in_dim, out_dim), scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _count("mlp")
        return self.proj(x) * self.scale


@GRAPH_CONVS.register("test_mean_conv", override=True)
class MeanConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        _count("conv")
        agg = torch.zeros_like(x).index_add_(0, edge_index[1], x[edge_index[0]])
        return self.linear(x + agg)


@FEEDFORWARDS.register("test_ffn", override=True)
class TinyFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _count("ffn")
        return self.net(x)


@NORMS.register("test_scale_norm", override=True)
class ScaleNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _count("norm")
        return self.g * x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)


@ATTENTIONS.register("test_no_attention", override=True)
class NoAttention(nn.Module):
    """Ignores context entirely; useful as an ablation baseline."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        _count("attention")
        return self.proj(x)


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    CALLS.clear()


def ring(n: int = 6, feat: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.arange(n)
    dst = (src + 1) % n
    return torch.randn(n, feat), torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], 1)


def test_gnn_slots_accept_custom_layers() -> None:
    x, edge_index = ring()
    model = GNNNodeClassifier(
        5,
        3,
        hidden_dim=8,
        conv="test_mean_conv",
        norm="test_scale_norm",
        activation="gelu",
        head={"name": "test_counting_mlp", "scale": 2.0},
    )
    assert model(x, edge_index).shape == (6, 3)
    assert CALLS == {"conv": 2, "norm": 1, "mlp": 1}


def test_transformer_slots_accept_custom_layers() -> None:
    model = TinyGPT(
        30,
        d_model=16,
        n_layers=2,
        n_heads=2,
        max_seq_len=8,
        attention="test_no_attention",
        ffn="test_ffn",
        norm="test_scale_norm",
    )
    assert model(torch.randint(30, (2, 5))).shape == (2, 5, 30)
    assert CALLS == {"attention": 2, "ffn": 2, "norm": 5}  # 2 per block + final norm


def test_builtin_alternatives_swap_without_custom_code() -> None:
    gpt = TinyGPT(30, d_model=16, n_layers=1, n_heads=2, ffn="gelu_mlp", norm="layernorm")
    assert gpt(torch.randint(30, (1, 4))).shape == (1, 4, 30)
    x, edge_index = ring()
    gnn = GNNNodeClassifier(
        5,
        3,
        hidden_dim=8,
        conv={"name": "gat", "heads": 2},
        norm="batchnorm",
        head={"name": "mlp", "hidden_dims": [8]},
    ).eval()
    assert gnn(x, edge_index).shape == (6, 3)


def test_actor_critic_networks_are_swappable() -> None:
    model = ActorCritic(4, 2, actor="test_counting_mlp", critic={"name": "mlp", "hidden_dims": [8]})
    dist, value = model(torch.randn(3, 4))
    assert dist.sample().shape == (3,) and value.shape == (3,)
    assert CALLS == {"mlp": 1}


def test_layers_swap_from_config_through_the_builder(tmp_path) -> None:  # type: ignore[no-untyped-def]
    trainer = (
        ExperimentBuilder("swap", output_dir=str(tmp_path))
        .with_data("synthetic_graph", num_nodes=60, feature_dim=5)
        .with_model(
            "gnn_node_classifier",
            hidden_dim=8,
            conv="test_mean_conv",
            head={"name": "test_counting_mlp"},
        )
        .with_objective("node_classification")
        .with_trainer("supervised", max_steps=2, batch_size=1, device="cpu", eval_every=0)
        .with_callbacks()
        .build()
    )
    trainer.fit()
    assert CALLS["conv"] > 0 and CALLS["mlp"] > 0


def test_unknown_layer_names_list_alternatives() -> None:
    with pytest.raises(RegistryError, match=r"Available:.*gcn"):
        GNNNodeClassifier(5, 3, conv="does_not_exist")


def test_optional_slots_and_role_registries() -> None:
    assert isinstance(build_layer(NORMS, None, dim=4), nn.Identity)
    for registry, name in (
        (ACTIVATIONS, "silu"),
        (NORMS, "rmsnorm"),
        (MLPS, "mlp"),
        (GRAPH_CONVS, "sage"),
        (FEEDFORWARDS, "swiglu"),
        (ATTENTIONS, "causal_self_attention"),
    ):
        assert name in registry
    assert "gnn_node_classifier" in MODELS
