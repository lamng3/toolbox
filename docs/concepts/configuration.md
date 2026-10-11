# Configuration

Configs are plain YAML composed by [Hydra](https://hydra.cc). The tree ships inside the package
(`src/toolbox/configs`), so `pip install` users get it too.

```
configs/
├── config.yaml        primary config: defaults list + name, seed, output_dir, callbacks
├── experiment/        complete experiments (`experiment=grpo_tiny`)
├── model/  data/  objective/  trainer/  optimizer/  scheduler/
├── tokenizer/  reward/  advantage/  env/  strategy/  launcher/
```

## Component specs

Every component section is a *spec*: a `name` that selects a registered factory, plus its
keyword arguments.

```yaml
objective:
  name: grpo          # → OBJECTIVES.get("grpo")
  clip_eps: 0.2       # → GRPOLoss(clip_eps=0.2, kl_coef=0.04)
  kl_coef: 0.04
```

There is no `_target_: package.module.Class` path, so configs survive refactors. The valid
names are in the [component catalog](components.md) and in `toolbox list`.

## Experiments

An experiment selects groups with `override` and then patches values:

```yaml title="configs/experiment/grpo_tiny.yaml"
# @package _global_
defaults:
  - override /model: tiny_gpt
  - override /data: prompts_arithmetic
  - override /objective: grpo
  - override /trainer: grpo
  - override /reward: exact_match
  - override /advantage: group_relative

name: grpo_tiny
trainer: {max_steps: 20, batch_size: 4, group_size: 4, max_new_tokens: 8}
optimizer: {lr: 1.0e-4}
```

## Overrides cheat-sheet

| Goal | Override |
| --- | --- |
| Swap a component | `objective=ipo`, `model=qwen2_5_0_5b_lora`, `strategy=fsdp` |
| Change a value | `trainer.max_steps=1000 optimizer.lr=1e-5` |
| Add a key that is not in the config | `+trainer.top_k=50` |
| Remove a key | `~scheduler` |
| Start from a checkpoint | `model.init_from=runs/sft/checkpoints/last.pt` |
| Resume a run | `trainer.resume_from=auto` |
| Your own configs first | `--config-dir path/to/configs` |
| Use your own components | `imports: [my_project.losses]` in the experiment file |

## Trainer settings

Loop-level settings live in `TrainerConfig` and are shared by every trainer:

| Key | Default | Meaning |
| --- | --- | --- |
| `max_steps` | 100 | optimizer steps |
| `batch_size` | 16 | per-device micro-batch (prompts for `grpo`) |
| `grad_accum_steps` | 1 | micro-batches per optimizer step |
| `grad_clip` | 1.0 | max global grad norm (0 = off) |
| `precision` | fp32 | `fp32` / `bf16` / `fp16` (autocast; GradScaler for fp16) |
| `device` | auto | `auto` → CUDA, then MPS, then CPU |
| `log_every` / `eval_every` / `save_every` | 10 / 50 / 0 | step intervals (0 = off) |
| `keep_last` | 2 | rolling `step_N.pt` checkpoints kept |
| `resume_from` | null | path or `auto` |

Keys that are not loop-level (`group_size`, `ppo_epochs`, `length_normalize`, ...) go to the
specific trainer's `from_builder`.
