"""Training objectives (Strategy pattern). Importing registers them in ``OBJECTIVES``."""

from toolbox.objectives.base import PolicyObjective, PreferenceObjective, SupervisedObjective
from toolbox.objectives.graph import (
    GraphContrastiveLoss,
    MaskedFeatureReconstruction,
    NodeClassificationLoss,
)
from toolbox.objectives.language import CausalLMLoss
from toolbox.objectives.policy import ClippedPolicyLoss, GRPOLoss, ReinforceLoss, kl_k3
from toolbox.objectives.preference import BradleyTerryLoss, DPOLoss, HingeLoss, IPOLoss
from toolbox.objectives.utils import masked_mean, sequence_logprobs, token_logprobs

__all__ = [
    "BradleyTerryLoss",
    "CausalLMLoss",
    "ClippedPolicyLoss",
    "DPOLoss",
    "GRPOLoss",
    "GraphContrastiveLoss",
    "HingeLoss",
    "IPOLoss",
    "MaskedFeatureReconstruction",
    "NodeClassificationLoss",
    "PolicyObjective",
    "PreferenceObjective",
    "ReinforceLoss",
    "SupervisedObjective",
    "kl_k3",
    "masked_mean",
    "sequence_logprobs",
    "token_logprobs",
]
