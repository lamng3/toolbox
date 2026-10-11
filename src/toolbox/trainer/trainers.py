"""Concrete trainers — one per training paradigm, all sharing :class:`BaseTrainer`'s loop.

Each trainer exposes ``from_builder`` so :class:`~toolbox.experiment.ExperimentBuilder` can
assemble it from config, while the constructor stays plain Python for programmatic use and
unit tests (pattern: *Dependency Injection*).
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from toolbox.core.constants import DEFAULT_CLIP_EPS, DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA
from toolbox.core.events import Event
from toolbox.core.registry import TRAINERS
from toolbox.core.types import Batch, LossOutput, Metrics
from toolbox.data.tokenization import Tokenizer
from toolbox.nn.models.base import CausalLM
from toolbox.objectives.base import PolicyObjective, PreferenceObjective, SupervisedObjective
from toolbox.objectives.policy import ClippedPolicyLoss
from toolbox.objectives.utils import sequence_logprobs
from toolbox.rewards import Reward
from toolbox.rl.advantage import AdvantageEstimator, compute_gae
from toolbox.rl.buffers import RolloutBuffer
from toolbox.rl.envs import Env
from toolbox.rl.rollout import SamplingConfig, TextRollout, rollout_logprobs
from toolbox.trainer.base import EVAL_PREFIX, BaseTrainer
from toolbox.trainer.config import TrainerConfig

if TYPE_CHECKING:
    from toolbox.experiment import ExperimentBuilder


def _expect(obj: object, kind: type, trainer: str) -> None:
    if not isinstance(obj, kind):
        raise TypeError(
            f"{trainer} needs a {kind.__name__}, got {type(obj).__name__}. "
            "Check the `objective` section of your config."
        )


def frozen_copy(model: nn.Module) -> nn.Module:
    """A detached, eval-mode copy used as the reference policy (DPO, GRPO KL)."""
    reference = copy.deepcopy(model).eval()
    reference.requires_grad_(False)
    return reference


def _base_kwargs(b: ExperimentBuilder, config: TrainerConfig) -> dict[str, Any]:
    """Dependencies every trainer takes; built lazily (and once) by the builder."""
    model = b.model()
    optimizer = b.optimizer(model)
    data = b.data()
    return {
        "model": model,
        "optimizer": optimizer,
        "config": config,
        "train_data": data.train if data else None,
        "eval_data": data.eval if data else None,
        "collate_fn": data.collate_fn if data else None,
        "scheduler": b.scheduler(optimizer, config.max_steps),
        "callbacks": b.callbacks(),
        "strategy": b.strategy(config.device),
        "run_config": b.run_config,
    }


# ------------------------------------------------------------------------------- supervised --
@TRAINERS.register("supervised")
class SupervisedTrainer(BaseTrainer):
    """Minimises a :class:`SupervisedObjective`: LM pretraining, SFT, reward models, GNNs,
    decision models."""

    def __init__(self, objective: SupervisedObjective, **kwargs: Any) -> None:
        _expect(objective, SupervisedObjective, "SupervisedTrainer")
        super().__init__(**kwargs)
        self.objective = objective

    @classmethod
    def from_builder(cls, b: ExperimentBuilder, config: TrainerConfig) -> SupervisedTrainer:
        return cls(objective=b.objective(), **_base_kwargs(b, config))

    def compute_loss(self, batch: Batch) -> LossOutput:
        return self.objective(self.model, batch)


# ------------------------------------------------------------------------------- preference --
@TRAINERS.register("preference")
class PreferenceTrainer(BaseTrainer):
    """Offline preference optimisation (DPO / IPO / SLiC) against a frozen reference model.

    Args:
        length_normalize: average instead of sum token log-probs (reduces length bias).
    """

    def __init__(
        self, objective: PreferenceObjective, length_normalize: bool = False, **kwargs: Any
    ) -> None:
        _expect(objective, PreferenceObjective, "PreferenceTrainer")
        reference = frozen_copy(kwargs["model"])
        super().__init__(**kwargs)
        self.objective = objective
        self.length_normalize = length_normalize
        self.reference = reference.to(self.device)

    @classmethod
    def from_builder(
        cls, b: ExperimentBuilder, config: TrainerConfig, length_normalize: bool = False
    ) -> PreferenceTrainer:
        return cls(
            objective=b.objective(), length_normalize=length_normalize, **_base_kwargs(b, config)
        )

    def _logps(self, model: nn.Module, batch: Batch, side: str) -> torch.Tensor:
        return sequence_logprobs(
            model,
            batch[f"{side}_input_ids"],
            batch[f"{side}_labels"],
            batch[f"{side}_attention_mask"],
            average=self.length_normalize,
        )

    def compute_loss(self, batch: Batch) -> LossOutput:
        with torch.no_grad():
            ref_chosen = self._logps(self.reference, batch, "chosen")
            ref_rejected = self._logps(self.reference, batch, "rejected")
        return self.objective(
            self._logps(self.model, batch, "chosen"),
            self._logps(self.model, batch, "rejected"),
            ref_chosen,
            ref_rejected,
        )


# ------------------------------------------------------------------------------------ GRPO --
@TRAINERS.register("grpo")
class GRPOTrainer(BaseTrainer):
    """Online RL for language models with verifiable or learned rewards.

    One step = sample ``group_size`` completions for ``batch_size`` prompts → score them →
    compute group-relative advantages → ``ppo_epochs`` clipped policy-gradient updates.
    Swap ``advantage`` (``group_relative`` / ``rloo`` / ``batch_normalized``) and ``objective``
    (``grpo`` / ``ppo_clip`` / ``reinforce``) in config to get GRPO, Dr. GRPO, RLOO, ...
    """

    def __init__(
        self,
        objective: PolicyObjective,
        tokenizer: Tokenizer,
        reward: Reward,
        advantage: AdvantageEstimator,
        sampling: SamplingConfig,
        ppo_epochs: int = 1,
        **kwargs: Any,
    ) -> None:
        _expect(objective, PolicyObjective, "GRPOTrainer")
        needs_reference = bool(getattr(objective, "requires_reference", False))
        reference = frozen_copy(kwargs["model"]) if needs_reference else None
        super().__init__(**kwargs)
        self.objective, self.tokenizer = objective, tokenizer
        self.reward, self.advantage = reward, advantage
        self.rollout = TextRollout(tokenizer, sampling)
        self.sampling = sampling
        self.ppo_epochs = ppo_epochs
        self.reference = reference.to(self.device) if reference is not None else None

    @classmethod
    def from_builder(
        cls,
        b: ExperimentBuilder,
        config: TrainerConfig,
        group_size: int = 4,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
        ppo_epochs: int = 1,
    ) -> GRPOTrainer:
        return cls(
            objective=b.objective(),
            tokenizer=b.tokenizer(),
            reward=b.reward(),
            advantage=b.advantage(),
            sampling=SamplingConfig(group_size, max_new_tokens, temperature, top_k),
            ppo_epochs=ppo_epochs,
            **_base_kwargs(b, config),
        )

    @property
    def policy(self) -> CausalLM:
        """The unwrapped policy, used for generation (DDP/FSDP wrappers don't expose it)."""
        return cast(CausalLM, self.unwrapped_model)

    def train_step(self) -> Metrics:
        batch = self.next_batch()
        rollout = self.rollout(self.policy, batch["prompt"], batch["answer"], self.device)
        rewards = self.reward(rollout.prompts, rollout.completions, rollout.references)
        advantages = self.advantage(rewards, rollout.group_size).to(self.device)
        mask = rollout.completion_mask
        with torch.no_grad(), self.precision.autocast():
            old_logprobs = rollout_logprobs(self.model, rollout)
            ref_logprobs = (
                rollout_logprobs(self.reference, rollout) if self.reference is not None else None
            )
        self.events.emit(Event.ROLLOUT_END, trainer=self, rollout=rollout, rewards=rewards)

        sums: dict[str, float] = defaultdict(float)
        for _ in range(self.ppo_epochs):
            with self.precision.autocast():
                logprobs = rollout_logprobs(self.model, rollout)
                out = self.objective(logprobs, old_logprobs, advantages, mask, ref_logprobs)
            sums["grad_norm"] += self.optimizer_step(out.loss) / self.ppo_epochs
            for key, value in out.detached_metrics().items():
                sums[key] += value / self.ppo_epochs
        sums.update(
            reward_mean=float(rewards.mean()),
            reward_std=float(rewards.std()) if rewards.numel() > 1 else 0.0,
            completion_tokens=float(mask.sum(-1).mean()),
        )
        return dict(sums)

    @torch.no_grad()
    def evaluate(self) -> Metrics:
        """Greedy-decode eval prompts and report the mean reward."""
        if self.eval_data is None:
            return {}
        greedy = TextRollout(
            self.tokenizer,
            SamplingConfig(
                group_size=1, max_new_tokens=self.sampling.max_new_tokens, temperature=0.0
            ),
        )
        rewards = []
        for i, batch in enumerate(self._loader(self.eval_data, shuffle=False)):
            if self.config.eval_batches and i >= self.config.eval_batches:
                break
            rollout = greedy(self.policy, batch["prompt"], batch["answer"], self.device)
            rewards.append(self.reward(rollout.prompts, rollout.completions, rollout.references))
        if not rewards:
            return {}
        return {f"{EVAL_PREFIX}reward": self.strategy.reduce_mean(float(torch.cat(rewards).mean()))}


# ------------------------------------------------------------------------- environment PPO --
@TRAINERS.register("ppo_env")
class EnvPPOTrainer(BaseTrainer):
    """PPO with an actor-critic on a (Gymnasium-style) environment — classic control, games,
    or any sequential decision problem exposed through :class:`~toolbox.rl.envs.Env`."""

    def __init__(
        self,
        env: Env,
        rollout_steps: int = 256,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        gamma: float = DEFAULT_GAMMA,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
        clip_eps: float = DEFAULT_CLIP_EPS,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.env = env
        self.rollout_steps, self.ppo_epochs, self.minibatch_size = (
            rollout_steps,
            ppo_epochs,
            minibatch_size,
        )
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.policy_loss = ClippedPolicyLoss(clip_eps=clip_eps)
        self.value_coef, self.entropy_coef = value_coef, entropy_coef
        self._obs, _ = env.reset(seed=seed)
        self._episode_return = 0.0
        self._finished_returns: list[float] = []

    @classmethod
    def from_builder(
        cls, b: ExperimentBuilder, config: TrainerConfig, **kwargs: Any
    ) -> EnvPPOTrainer:
        return cls(env=b.env(), seed=b.seed, **kwargs, **_base_kwargs(b, config))

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _collect(self) -> RolloutBuffer:
        buffer = RolloutBuffer()
        model = self.unwrapped_model
        for _ in range(self.rollout_steps):
            obs = self._obs_tensor(self._obs)
            dist, value = model(obs)
            action = dist.sample()
            next_obs, reward, terminated, truncated, _ = self.env.step(int(action))
            done = terminated or truncated
            buffer.add(obs, int(action), float(dist.log_prob(action)), float(value), reward, done)
            self._episode_return += reward
            if done:
                self._finished_returns.append(self._episode_return)
                self._episode_return = 0.0
                next_obs, _ = self.env.reset()
            self._obs = next_obs
        _, last_value = model(self._obs_tensor(self._obs))
        data = buffer.tensors()
        buffer.advantages, buffer.returns = compute_gae(
            data["rewards"],
            data["values"],
            data["dones"],
            last_value.cpu(),
            self.gamma,
            self.gae_lambda,
        )
        return buffer

    def train_step(self) -> Metrics:
        buffer = self._collect()
        sums: dict[str, float] = defaultdict(float)
        updates = 0
        for _ in range(self.ppo_epochs):
            for mb in buffer.minibatches(self.minibatch_size):
                mb = {k: v.to(self.device) for k, v in mb.items()}
                dist, values = self.model(mb["obs"])
                logprobs = dist.log_prob(mb["actions"])
                ones = torch.ones_like(logprobs)
                policy = self.policy_loss(
                    logprobs[:, None], mb["logprobs"][:, None], mb["advantages"], ones[:, None]
                )
                value_loss = F.mse_loss(values, mb["returns"])
                entropy = dist.entropy().mean()
                loss = policy.loss + self.value_coef * value_loss - self.entropy_coef * entropy
                sums["grad_norm"] += self.optimizer_step(loss)
                sums["policy_loss"] += policy.loss.item()
                sums["value_loss"] += value_loss.item()
                sums["entropy"] += entropy.item()
                updates += 1
        metrics = {k: v / max(updates, 1) for k, v in sums.items()}
        if self._finished_returns:
            metrics["episode_return"] = float(np.mean(self._finished_returns[-10:]))
        return metrics


__all__ = [
    "EnvPPOTrainer",
    "GRPOTrainer",
    "PreferenceTrainer",
    "SupervisedTrainer",
    "frozen_copy",
]
