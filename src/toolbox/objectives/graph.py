"""Graph objectives: supervised node classification and self-supervised pretraining."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from toolbox.core.registry import OBJECTIVES
from toolbox.core.types import Batch, LossOutput
from toolbox.objectives.base import SupervisedObjective

TRAIN_MASK = "train_mask"


@OBJECTIVES.register("node_classification")
class NodeClassificationLoss(SupervisedObjective):
    """Cross-entropy over nodes selected by ``mask_key`` (``train_mask`` by default)."""

    def __init__(self, mask_key: str = TRAIN_MASK) -> None:
        self.mask_key = mask_key

    def __call__(self, model: nn.Module, batch: Batch) -> LossOutput:
        logits = model(batch["x"], batch["edge_index"])
        mask = batch[self.mask_key]
        loss = F.cross_entropy(logits[mask], batch["y"][mask])
        accuracy = (logits[mask].argmax(-1) == batch["y"][mask]).float().mean()
        return LossOutput(loss, {"accuracy": float(accuracy)})


@OBJECTIVES.register("masked_feature")
class MaskedFeatureReconstruction(SupervisedObjective):
    """GraphMAE (Hou et al., 2022): mask node features, reconstruct them with a scaled cosine
    error ``(1 − cos(x̂, x))^γ`` on the masked nodes only.

    Requires a model exposing ``encoder``, ``decoder`` and ``mask_token``
    (:class:`~toolbox.nn.models.gnn.GNNPretrainer`).
    """

    def __init__(self, mask_rate: float = 0.5, gamma: float = 2.0) -> None:
        self.mask_rate, self.gamma = mask_rate, gamma

    def __call__(self, model: Any, batch: Batch) -> LossOutput:
        x, edge_index = batch["x"], batch["edge_index"]
        masked = torch.rand(x.shape[0], device=x.device) < self.mask_rate
        x_in = torch.where(masked[:, None], model.mask_token.expand_as(x), x)
        recon = model.decoder(model.encoder(x_in, edge_index))
        cos = F.cosine_similarity(recon[masked], x[masked], dim=-1)
        loss = (1 - cos).pow(self.gamma).mean()
        return LossOutput(loss, {"cosine": cos.mean().item()})


def drop_edges(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= p
    return edge_index[:, keep]


def mask_features(x: torch.Tensor, p: float) -> torch.Tensor:
    keep = (torch.rand(x.shape[1], device=x.device) >= p).to(x.dtype)
    return x * keep


@OBJECTIVES.register("graph_contrastive")
class GraphContrastiveLoss(SupervisedObjective):
    """GRACE (Zhu et al., 2020): two stochastic views per graph, node-level InfoNCE where each
    node's counterpart in the other view is the positive and all other nodes are negatives."""

    def __init__(
        self, temperature: float = 0.5, edge_drop: float = 0.2, feature_mask: float = 0.3
    ) -> None:
        self.temperature, self.edge_drop, self.feature_mask = temperature, edge_drop, feature_mask

    def _view(self, model: Any, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_view = mask_features(x, self.feature_mask)
        h = model.encoder(x_view, drop_edges(edge_index, self.edge_drop))
        return F.normalize(model.projector(h), dim=-1)

    def __call__(self, model: nn.Module, batch: Batch) -> LossOutput:
        z1 = self._view(model, batch["x"], batch["edge_index"])
        z2 = self._view(model, batch["x"], batch["edge_index"])
        logits = z1 @ z2.T / self.temperature
        targets = torch.arange(z1.shape[0], device=z1.device)
        loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
        accuracy = (logits.argmax(-1) == targets).float().mean()
        return LossOutput(loss, {"contrastive_accuracy": float(accuracy)})
