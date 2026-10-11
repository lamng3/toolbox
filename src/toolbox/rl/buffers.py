"""On-policy trajectory buffer for environment RL."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from toolbox.core.constants import EPS


@dataclass
class RolloutBuffer:
    """Stores one rollout of ``T`` steps, then yields shuffled minibatches for PPO epochs."""

    obs: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        logprob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(float(done))

    def __len__(self) -> int:
        return len(self.actions)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.stack(self.obs),
            "actions": torch.tensor(self.actions),
            "logprobs": torch.tensor(self.logprobs),
            "values": torch.tensor(self.values),
            "rewards": torch.tensor(self.rewards),
            "dones": torch.tensor(self.dones),
        }

    def minibatches(
        self, batch_size: int, normalize_advantages: bool = True
    ) -> Iterator[dict[str, torch.Tensor]]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("compute advantages before sampling minibatches")
        data = self.tensors()
        advantages = self.advantages
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
        data.update(advantages=advantages, returns=self.returns)
        perm = torch.randperm(len(self))
        for start in range(0, len(self), batch_size):
            idx = perm[start : start + batch_size]
            yield {k: v[idx] for k, v in data.items()}
