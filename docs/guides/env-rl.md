# Environment RL

`ppo_env` is a compact PPO implementation (actor-critic, GAE, clipped surrogate, value loss,
entropy bonus) for sequential decision problems behind the `Env` interface:

```python
class Env:
    obs_dim: int
    num_actions: int
    def reset(self, seed=None) -> tuple[obs, info]: ...
    def step(self, action) -> tuple[obs, reward, terminated, truncated, info]: ...
```

```bash
toolbox train experiment=ppo_cartpole trainer.max_steps=200      # built-in CartPole, no deps
uv pip install -e ".[rl]"
toolbox train experiment=ppo_cartpole env=gymnasium env.env_id=Acrobot-v1
```

`obs_dim` and `num_actions` are injected from the environment into `actor_critic`.

To use your own simulator (a scheduling problem, a network-control task, a game), subclass
`Env`, register it with `@ENVS.register("my_env")`, and set `env.name=my_env`.

PPO reuses the same `ClippedPolicyLoss` as LLM RL. Only the advantage estimator (GAE with a
learned critic) and the rollout source differ.
