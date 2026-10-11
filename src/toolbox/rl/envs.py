"""Environments for classic (non-text) RL.

:class:`Env` is a Gymnasium-compatible subset, so built-in envs and any ``gymnasium`` env (via
:class:`GymnasiumAdapter`) are interchangeable (patterns: *Polymorphism*, *Adapter*).
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from toolbox.core.decorators import require
from toolbox.core.registry import ENVS

StepResult = tuple[np.ndarray, float, bool, bool, dict[str, Any]]


class Env(ABC):
    """``reset() -> (obs, info)``; ``step(a) -> (obs, reward, terminated, truncated, info)``."""

    obs_dim: int
    num_actions: int

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]: ...

    @abstractmethod
    def step(self, action: int) -> StepResult: ...


@ENVS.register("cartpole")
class CartPole(Env):
    """Classic cart-pole balancing (Barto et al., 1983), dependency-free.

    Physics constants follow Gymnasium's ``CartPole-v1``; reward +1 per step balanced.
    """

    GRAVITY, CART_MASS, POLE_MASS, POLE_HALF_LENGTH = 9.8, 1.0, 0.1, 0.5
    FORCE, DT = 10.0, 0.02
    ANGLE_LIMIT = 12 * 2 * math.pi / 360
    POSITION_LIMIT = 2.4
    INIT_NOISE = 0.05

    obs_dim = 4
    num_actions = 2

    def __init__(self, max_steps: int = 500) -> None:
        self.max_steps = max_steps
        self._rng = random.Random()
        self.state: np.ndarray = np.zeros(4)
        self.steps = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)
        self.state = np.array(
            [self._rng.uniform(-self.INIT_NOISE, self.INIT_NOISE) for _ in range(4)]
        )
        self.steps = 0
        return self.state.astype(np.float32), {}

    def step(self, action: int) -> StepResult:
        x, x_dot, theta, theta_dot = self.state
        force = self.FORCE if action == 1 else -self.FORCE
        total_mass = self.CART_MASS + self.POLE_MASS
        pole_moment = self.POLE_MASS * self.POLE_HALF_LENGTH
        cos, sin = math.cos(theta), math.sin(theta)
        temp = (force + pole_moment * theta_dot**2 * sin) / total_mass
        theta_acc = (self.GRAVITY * sin - cos * temp) / (
            self.POLE_HALF_LENGTH * (4.0 / 3.0 - self.POLE_MASS * cos**2 / total_mass)
        )
        x_acc = temp - pole_moment * theta_acc * cos / total_mass
        x, x_dot = x + self.DT * x_dot, x_dot + self.DT * x_acc
        theta, theta_dot = theta + self.DT * theta_dot, theta_dot + self.DT * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        terminated = bool(abs(x) > self.POSITION_LIMIT or abs(theta) > self.ANGLE_LIMIT)
        truncated = self.steps >= self.max_steps
        return self.state.astype(np.float32), 1.0, terminated, truncated, {}


@ENVS.register("gymnasium")
class GymnasiumAdapter(Env):
    """Wrap any discrete-action ``gymnasium`` environment by id."""

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        gym = require("gymnasium", extra="rl")
        self._env = gym.make(env_id, **kwargs)
        self.obs_dim = int(np.prod(self._env.observation_space.shape))
        self.num_actions = int(self._env.action_space.n)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed)
        return np.asarray(obs, dtype=np.float32).ravel(), info

    def step(self, action: int) -> StepResult:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return np.asarray(obs, dtype=np.float32).ravel(), float(reward), terminated, truncated, info
