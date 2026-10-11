"""Graph neural networks for self-supervised pretraining and supervised fine-tuning.

The workflow mirrors LLMs: *pretrain* a :class:`GNNEncoder` with a self-supervised objective
(masked feature reconstruction or graph contrastive learning) using :class:`GNNPretrainer`, then
*post-train* a :class:`GNNNodeClassifier` initialised from the pretrained encoder weights.

Every building block is a swappable layer spec: the message-passing layer (``conv``, from
``GRAPH_CONVS``), the ``activation``, an optional ``norm`` between layers, and the output
``head`` / ``decoder`` / ``projector`` (from ``MLPS``). For example::

    model:
      name: gnn_node_classifier
      conv: {name: gat, heads: 8}
      norm: layernorm
      head: {name: mlp, hidden_dims: [64], dropout: 0.2}
"""

from __future__ import annotations

import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from toolbox.core.registry import ACTIVATIONS, GRAPH_CONVS, MLPS, MODELS, NORMS
from toolbox.nn.layers.common import LayerSpec, build_layer

ENCODER_PREFIX = "encoder."


@MODELS.register("gnn_encoder")
class GNNEncoder(nn.Module):
    """Stack of ``conv → norm → activation → dropout`` blocks (the last conv is bare).

    Args:
        conv: graph layer spec, e.g. ``"gcn"``, ``"sage"``, ``{"name": "gat", "heads": 8}``,
            or any layer you registered in ``GRAPH_CONVS``.
        activation: activation spec (``ACTIVATIONS``).
        norm: optional norm spec between layers (``NORMS``), e.g. ``"layernorm"``.
        out_dim: width of the last layer (defaults to ``hidden_dim``).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        conv: LayerSpec = "gcn",
        dropout: float = 0.5,
        activation: LayerSpec = "relu",
        norm: LayerSpec | None = None,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        out_dim = out_dim or hidden_dim
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            build_layer(GRAPH_CONVS, conv, in_dim=a, out_dim=b) for a, b in itertools.pairwise(dims)
        )
        hidden_blocks = num_layers - 1
        self.norms = nn.ModuleList(
            build_layer(NORMS, norm, dim=hidden_dim) for _ in range(hidden_blocks)
        )
        self.activation = build_layer(ACTIVATIONS, activation)
        self.dropout = dropout
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.norms):
                x = self.activation(self.norms[i](x))
                x = F.dropout(x, self.dropout, self.training)
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
        conv: LayerSpec = "gcn",
        dropout: float = 0.5,
        proj_dim: int = 64,
        activation: LayerSpec = "relu",
        norm: LayerSpec | None = None,
        decoder: LayerSpec = "linear",
        projector: LayerSpec | None = None,
    ) -> None:
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden_dim, num_layers, conv, dropout, activation, norm)
        self.decoder = build_layer(MLPS, decoder, in_dim=hidden_dim, out_dim=in_dim)
        projector = projector or {"name": "mlp", "hidden_dims": [hidden_dim], "activation": "gelu"}
        self.projector = build_layer(MLPS, projector, in_dim=hidden_dim, out_dim=proj_dim)
        self.mask_token = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)


@MODELS.register("gnn_node_classifier")
class GNNNodeClassifier(nn.Module):
    """Encoder → activation → head, or with ``head=None`` the classic Kipf & Welling GNN.

    * ``head="linear"`` (default) / any ``MLPS`` spec: ``L`` convs to ``hidden_dim``, then a
      separate head. This is the layout to use with pretrained encoders.
    * ``head=None``: the last conv outputs the class logits directly, e.g. the standard 2-layer
      GCN ``conv → relu → conv`` (Kipf & Welling, 2017; PyG's example ``GCN``).

    Args:
        conv: graph layer spec for the encoder (see :class:`GNNEncoder`).
        activation: activation spec, used in the encoder and before the head.
        norm: optional norm spec between encoder layers.
        head: classification head spec (``MLPS``), ``"linear"`` or e.g.
            ``{"name": "mlp", "hidden_dims": [64]}``; ``None`` for the classic layout.
        pretrained_checkpoint: a toolbox checkpoint (``.pt``) of a :class:`GNNPretrainer`;
            only the ``encoder.*`` weights are loaded.
        freeze_encoder: linear-probe evaluation, which trains only the head.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        conv: LayerSpec = "gcn",
        dropout: float = 0.5,
        activation: LayerSpec = "relu",
        norm: LayerSpec | None = None,
        head: LayerSpec | None = "linear",
        pretrained_checkpoint: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        classic = head is None
        self.encoder = GNNEncoder(
            in_dim,
            hidden_dim,
            num_layers,
            conv,
            dropout,
            activation,
            norm,
            out_dim=num_classes if classic else hidden_dim,
        )
        self.activation = nn.Identity() if classic else build_layer(ACTIVATIONS, activation)
        self.head = build_layer(MLPS, head, in_dim=hidden_dim, out_dim=num_classes)
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
        return self.head(self.activation(self.encoder(x, edge_index)))
