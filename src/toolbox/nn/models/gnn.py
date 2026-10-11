"""Graph neural networks for self-supervised pretraining and supervised fine-tuning.

The workflow mirrors LLMs: *pretrain* a :class:`GNNEncoder` with a self-supervised objective
(masked feature reconstruction or graph contrastive learning) using :class:`GNNPretrainer`, then
*post-train* a :class:`GNNNodeClassifier` initialised from the pretrained encoder weights.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from toolbox.core.registry import MODELS
from toolbox.nn.layers import MLP
from toolbox.nn.layers.graph import GRAPH_CONVS

ENCODER_PREFIX = "encoder."


@MODELS.register("gnn_encoder")
class GNNEncoder(nn.Module):
    """Stack of message-passing layers: ``conv → activation → dropout`` (last layer is bare)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        conv: str = "gcn",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        conv_cls = GRAPH_CONVS[conv]
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(conv_cls(a, b) for a, b in itertools.pairwise(dims))
        self.dropout = dropout
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.dropout(F.relu(x), self.dropout, self.training)
        return x


@MODELS.register("gnn_pretrainer")
class GNNPretrainer(nn.Module):
    """Encoder + heads used only during self-supervised pretraining.

    * ``decoder`` reconstructs masked node features (GraphMAE-style).
    * ``projector`` maps embeddings to the contrastive space (GRACE-style).
    * ``mask_token`` replaces features of masked nodes.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        conv: str = "gcn",
        dropout: float = 0.5,
        proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden_dim, num_layers, conv, dropout)
        self.decoder = nn.Linear(hidden_dim, in_dim)
        self.projector = MLP(hidden_dim, [hidden_dim], proj_dim, activation="gelu")
        self.mask_token = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)


@MODELS.register("gnn_node_classifier")
class GNNNodeClassifier(nn.Module):
    """Encoder + linear head; can start from a pretrained encoder checkpoint.

    Args:
        pretrained_checkpoint: a toolbox checkpoint (``.pt``) of a :class:`GNNPretrainer`;
            only the ``encoder.*`` weights are loaded.
        freeze_encoder: linear-probe evaluation — train only the classification head.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        conv: str = "gcn",
        dropout: float = 0.5,
        pretrained_checkpoint: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden_dim, num_layers, conv, dropout)
        self.head = nn.Linear(hidden_dim, num_classes)
        if pretrained_checkpoint:
            self.load_encoder(pretrained_checkpoint)
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def load_encoder(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        encoder_state = {
            k.removeprefix(ENCODER_PREFIX): v
            for k, v in state.items()
            if k.startswith(ENCODER_PREFIX)
        }
        if not encoder_state:
            raise ValueError(f"No '{ENCODER_PREFIX}*' weights found in {path}")
        self.encoder.load_state_dict(encoder_state)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(F.relu(self.encoder(x, edge_index)))
