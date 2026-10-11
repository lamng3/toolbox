# Extending the toolbox

Every technique is a registered component. To add one:

1. **Implement the interface** for its family (`SupervisedObjective`, `PreferenceObjective`,
   `PolicyObjective`, `Reward`, `AdvantageEstimator`, `Callback`, `InferenceEngine`, `Env`,
   `Tokenizer`, `CausalLM`, ...).
2. **Register** it with the matching registry decorator.
3. **Select** it from config by name.

```python
from toolbox.core.registry import REWARDS
from toolbox.rewards import Reward
import torch

@REWARDS.register("contains_keyword")
class ContainsKeyword(Reward):
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

    def __call__(self, prompts, completions, references) -> torch.Tensor:
        return torch.tensor([float(self.keyword in c) for c in completions])
```

```python
from toolbox.experiment import ExperimentBuilder, load_config

builder = ExperimentBuilder.from_config(load_config(["experiment=grpo_tiny"]))
builder.with_reward("contains_keyword", keyword="because").build().fit()
```

For the CLI, add a config file for it (`configs/reward/contains_keyword.yaml` in your
`--config-dir`) and make sure its module is imported. The simplest way is to ship it as a
plugin (see below); then `toolbox train experiment=grpo_tiny reward=contains_keyword` works.

## Registries

| Registry | Interface | Examples |
| --- | --- | --- |
| `MODELS` | `nn.Module` (`CausalLM` for LMs) | `tiny_gpt`, `hf_causal_lm`, `gnn_encoder` |
| `TOKENIZERS` | `Tokenizer` | `byte`, `hf` |
| `DATASETS` | factory → `DataBundle` | `sft`, `preference`, `prompts`, `decisions`, `traces` |
| `OBJECTIVES` | objective families | `sft`, `dpo`, `grpo`, `decision_ce` |
| `REWARDS` | `Reward` | `exact_match`, `llm_judge` |
| `ADVANTAGES` | `AdvantageEstimator` | `group_relative`, `rloo` |
| `TRAINERS` | `BaseTrainer` + `from_builder` | `supervised`, `preference`, `grpo`, `ppo_env` |
| `CALLBACKS` | `Callback` | `wandb`, `torch_profiler`, `trace_rollouts` |
| `ENGINES` | `InferenceEngine` | `sglang`, `vllm`, `openai_http` |
| `ENVS` | `Env` | `cartpole`, `gymnasium` |
| `STRATEGIES` | `Strategy` | `single`, `ddp`, `fsdp` |
| `LAUNCHERS` | `LaunchCommand` | `local`, `torchrun`, `slurm` |

## A new training paradigm

Subclass `BaseTrainer`. Override `compute_loss` if a step is "loss of a batch", or
`train_step` if it is more involved (rollouts, multiple updates). Then add `from_builder`:

```python
@TRAINERS.register("my_trainer")
class MyTrainer(BaseTrainer):
    def __init__(self, objective, my_param: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.objective, self.my_param = objective, my_param

    @classmethod
    def from_builder(cls, b, config, my_param: float = 1.0):
        from toolbox.trainer.trainers import _base_kwargs
        return cls(objective=b.objective(), my_param=my_param, **_base_kwargs(b, config))

    def train_step(self):
        ...
        return {"loss": loss_value, "grad_norm": self.optimizer_step(loss)}
```

You get logging, evaluation, checkpointing, resuming, mixed precision, DDP and SLURM launching
without writing them.

## Shipping it as a plugin

Put your components in a package and advertise the module:

```toml title="pyproject.toml of your package"
[project.entry-points."toolbox.plugins"]
my_lab = "my_lab.toolbox_components"
```

After `pip install my-lab`, `toolbox list` shows your components and every config can use
them. Add your own experiment YAMLs with `--config-dir`.
