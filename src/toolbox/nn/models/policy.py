"""Actor-critic networks for environment-based RL (PPO).

The ``actor`` and ``critic`` networks are ``MLPS`` specs, so a custom network registered with
``@MLPS.register(...)`` can replace either one.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from toolbox.core.registry import MLPS, MODELS
from toolbox.nn.layers import LayerSpec, build_layer


@MODELS.register("actor_critic")
class ActorCritic(nn.Module):
    """Separate MLP policy (categorical actions) and value networks over a flat observation."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        activation: LayerSpec = "tanh",
        actor: LayerSpec | None = None,
        critic: LayerSpec | None = None,
    ) -> None:
        super().__init__()
        default = {"name": "mlp", "hidden_dims": list(hidden_dims), "activation": activation}
        self.actor = build_layer(MLPS, actor or default, in_dim=obs_dim, out_dim=num_actions)
        self.critic = build_layer(MLPS, critic or default, in_dim=obs_dim, out_dim=1)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        return self.distribution(obs), self.value(obs)
