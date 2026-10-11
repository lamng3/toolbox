"""Model zoo. Importing this package registers every built-in model in ``MODELS``."""

from toolbox.nn.models.base import CausalLM, sample_next_token
from toolbox.nn.models.gnn import GNNEncoder, GNNNodeClassifier, GNNPretrainer
from toolbox.nn.models.gpt import SequenceRewardModel, TinyGPT
from toolbox.nn.models.hf import HFCausalLM
from toolbox.nn.models.policy import ActorCritic

__all__ = [
    "ActorCritic",
    "CausalLM",
    "GNNEncoder",
    "GNNNodeClassifier",
    "GNNPretrainer",
    "HFCausalLM",
    "SequenceRewardModel",
    "TinyGPT",
    "sample_next_token",
]
