"""Actor-critic networks for environment-based RL (PPO)."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from toolbox.core.registry import MODELS
from toolbox.nn.layers import MLP


@MODELS.register("actor_critic")
class ActorCritic(nn.Module):
    """Separate MLP policy (categorical actions) and value networks over a flat observation."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.actor = MLP(obs_dim, hidden_dims, num_actions, activation=activation)
        self.critic = MLP(obs_dim, hidden_dims, 1, activation=activation)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        return self.distribution(obs), self.value(obs)
