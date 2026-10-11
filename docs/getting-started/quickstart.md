# Quickstart

Every packaged experiment has a *tiny* version that runs on a laptop CPU in seconds. Use these
to check that a pipeline is wired correctly before spending GPU hours.

## 1. Train something

```bash
toolbox train experiment=sft_tiny
```

```
INFO  Starting SupervisedTrainer for 200 steps → runs/sft_tiny
INFO  step 10   loss=2.91  perplexity=18.3  grad_norm=1.2  lr=0.001 ...
INFO  eval @ 50 eval/loss=0.88 eval/perplexity=2.41
```

A run directory looks like this:

```
runs/sft_tiny/
├── config.yaml          # fully resolved config: reproduce with `toolbox train --config-dir ...`
├── metrics.jsonl        # one line per step / eval
└── checkpoints/
    ├── last.pt
    └── step_200.pt
```

## 2. Override anything

Overrides use [Hydra](https://hydra.cc) syntax: `group=option` swaps a component, and
`key.path=value` changes a value.

```bash
toolbox train experiment=sft_tiny trainer.max_steps=50 optimizer.lr=3e-4 model.n_layers=4
toolbox train experiment=dpo_tiny objective=ipo                  # swap the loss
toolbox train experiment=grpo_tiny advantage=rloo objective=reinforce
toolbox config experiment=grpo_tiny                               # print the composed config
toolbox list objective                                            # what's registered
```

Every name and preset is also listed in the [component catalog](../concepts/components.md).

## 3. Chain post-training stages

`model.init_from` starts a stage from the previous stage's checkpoint:

```bash
toolbox train experiment=sft_tiny
toolbox train experiment=grpo_tiny model.init_from=runs/sft_tiny/checkpoints/last.pt
```

[`examples/recipes/rlhf_pipeline.sh`](../examples/recipes.md) runs the whole thing: SFT → reward model → DPO and GRPO.

## 4. Use it from Python

```python
from toolbox.experiment import ExperimentBuilder

trainer = (
    ExperimentBuilder("my-grpo")
    .with_tokenizer("byte")
    .with_data("prompts", path="builtin://arithmetic.jsonl")
    .with_model("tiny_gpt", d_model=64, n_layers=2, init_from="runs/sft_tiny/checkpoints/last.pt")
    .with_objective("grpo", kl_coef=0.02)
    .with_reward("exact_match")
    .with_advantage("group_relative")
    .with_trainer("grpo", max_steps=50, batch_size=4, group_size=8, max_new_tokens=8)
    .with_callbacks("console", "jsonl", "trace_rollouts")
    .build()
)
metrics = trainer.fit()
```

## 5. Go to a real model

The `*_qwen*` experiments use `Qwen/Qwen2.5-0.5B-Instruct` (`pip install -e ".[llm]"`, one GPU):

```bash
toolbox train experiment=sft_qwen data.path=data/processed/my_sft.jsonl
toolbox train experiment=grpo_qwen_gsm8k
```

Any Hugging Face model works: `model.model_name_or_path=meta-llama/Llama-3.2-1B-Instruct
tokenizer.name_or_path=meta-llama/Llama-3.2-1B-Instruct`.

## Packaged experiments

| Experiment | Trainer | What it does | Hardware |
| --- | --- | --- | --- |
| `sft_tiny` | supervised | SFT on arithmetic Q/A | CPU |
| `dpo_tiny` | preference | DPO on correct/incorrect answer pairs | CPU |
| `reward_model_tiny` | supervised | Bradley–Terry reward model | CPU |
| `grpo_tiny` | grpo | GRPO with exact-match reward, traces + memory | CPU |
| `decision_tiny` / `decision_rl_tiny` | supervised | decision model via CE / policy gradient | CPU |
| `sft_from_traces` | supervised | expert iteration on traced rollouts | CPU |
| `ppo_cartpole` | ppo_env | PPO actor-critic on CartPole | CPU |
| `gnn_pretrain` → `gnn_finetune` | supervised | GraphMAE pretraining → node classification | CPU |
| `lm_pretrain_tiny` | supervised | minimal next-token pretraining | CPU |
| `sft_qwen`, `dpo_qwen`, `grpo_qwen_gsm8k` | — | the same recipes on Qwen2.5-0.5B (+LoRA) | GPU |

Next: the [research workflow](research.md) shows how to go from these baselines to your own
idea.
