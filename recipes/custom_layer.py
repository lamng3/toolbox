"""Swap model internals for your own layers, without touching toolbox code.

1. A custom transformer feed-forward (ReLU², from Primer) replaces SwiGLU in ``tiny_gpt``.
2. A custom residual head replaces the linear head of a GNN node classifier.

Each is trained next to the built-in it replaces, so the comparison is one line of config.

    python recipes/custom_layer.py
"""

from __future__ import annotations

import torch
from torch import nn

from toolbox.core.registry import FEEDFORWARDS, MLPS
from toolbox.experiment import ExperimentBuilder

STEPS = 60


@FEEDFORWARDS.register("relu2")
class ReLUSquared(nn.Module):
    """Primer-style feed-forward: ``W2(relu(W1 x)^2)`` (So et al., 2021)."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1, self.w2 = nn.Linear(d_model, hidden_dim), nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.relu(self.w1(x)) ** 2)


@MLPS.register("residual_head")
class ResidualHead(nn.Module):
    """A classification head with one residual hidden layer."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.inner = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, in_dim))
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x + self.inner(x))


def language_model(ffn: str) -> float:
    trainer = (
        ExperimentBuilder(f"custom_layer_lm_{ffn}", output_dir=f"runs/custom_layer/lm_{ffn}")
        .with_tokenizer("byte")
        .with_data("text_lm", block_size=64)
        .with_model("tiny_gpt", d_model=64, n_layers=2, n_heads=4, ffn=ffn)
        .with_objective("lm_ce")
        .with_optimizer("adamw", lr=1e-3)
        .with_trainer("supervised", max_steps=STEPS, batch_size=8, device="cpu", eval_every=0)
        .with_callbacks()
        .build()
    )
    return float(trainer.fit()["eval/loss"])


def gnn(head: str) -> float:
    trainer = (
        ExperimentBuilder(f"custom_layer_gnn_{head}", output_dir=f"runs/custom_layer/gnn_{head}")
        .with_data("synthetic_graph")
        .with_model("gnn_node_classifier", hidden_dim=32, head=head)
        .with_objective("node_classification")  # train on train_mask, eval on val_mask
        .with_optimizer("adam", lr=1e-2, weight_decay=5e-4)
        .with_trainer("supervised", max_steps=STEPS, batch_size=1, device="cpu", eval_every=0)
        .with_callbacks()
        .build()
    )
    return float(trainer.fit()["eval/accuracy"])


def main() -> None:
    for ffn in ("swiglu", "relu2"):
        print(f"tiny_gpt ffn={ffn:8s} eval loss = {language_model(ffn):.3f}")
    for head in ("linear", "residual_head"):
        print(f"gnn head={head:14s} eval accuracy = {gnn(head):.3f}")


if __name__ == "__main__":
    main()
