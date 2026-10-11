# toolbox

toolbox is a Python library and template for getting from an idea to results quickly in
**post-training**: SFT, DPO, GRPO and other RL methods for language models. Around that core it
also provides AnyJev-style **decision models**, **agent tracing** (log every action, then train
on the log), memory-driven self-improvement, GNN pre-/post-training, and **SGLang / vLLM**
serving for inference work. Spin up a GPU VM or an HPC allocation and the same command runs.

Every technique is a component behind one small interface. You register it by name and pick
it from config. The code base is also written to be studied: each of the classic refactoring
and design patterns is used where it solves a real problem, and each is
[mapped to the code](https://lamng3.github.io/toolbox/concepts/design-patterns/).

[Documentation](https://lamng3.github.io/toolbox/)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![CI](https://github.com/lamng3/toolbox/actions/workflows/ci.yml/badge.svg)](https://github.com/lamng3/toolbox/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)

## Quick start

```bash
git clone https://github.com/lamng3/toolbox.git
cd toolbox
uv venv --python 3.11 && uv pip install -e ".[dev]"

toolbox train experiment=sft_tiny                                  # CPU, ~10 s
toolbox train experiment=grpo_tiny model.init_from=runs/sft_tiny/checkpoints/last.pt
```

On a GPU, with a real model:

```bash
uv pip install -e ".[llm]"
toolbox train experiment=grpo_qwen_gsm8k                           # Qwen2.5-0.5B + LoRA, GRPO on GSM8K
```

On a SLURM cluster (look at the script with `--dry-run`, then drop the flag to submit):

```bash
toolbox launch experiment=grpo_qwen_gsm8k launcher=slurm \
    launcher.partition=gpu launcher.gpus_per_node=4 --dry-run
```

From Python:

```python
from toolbox.experiment import ExperimentBuilder

trainer = (
    ExperimentBuilder("dpo-demo")
    .with_tokenizer("byte")
    .with_data("preference", path="builtin://preference.jsonl")
    .with_model("tiny_gpt", d_model=64, n_layers=2)
    .with_objective("dpo", beta=0.1)
    .with_trainer("preference", max_steps=100, batch_size=16)
    .build()
)
trainer.fit()
```

## What's supported

| Area | Components |
| --- | --- |
| **Post-training objectives** | `sft`, `dpo`, `ipo`, `slic`, `reward_bt` (reward models), `grpo`, `ppo_clip`, `reinforce` |
| **Online RL** | group-relative (GRPO), Dr. GRPO, RLOO and batch-normalised advantages; k3 KL to a frozen reference; DAPO-style asymmetric clipping |
| **Rewards** | `exact_match`, `format`, `length_penalty`, `weighted_sum` (composite), learned `reward_model`, `llm_judge` (any inference engine) |
| **Models** | any Hugging Face causal LM (+ LoRA via PEFT); `tiny_gpt` (Llama-style, from scratch); reward model; GCN / GraphSAGE / GAT; actor-critic |
| **Decision models** | typed questions → option probabilities read from next-token logits, debiased over option rotations; trained by CE, self-distillation or policy gradient; flip-rate eval |
| **Agent tracing** | `ActionRecord` for decisions, tool calls and generations; nested spans, `@traced`, delayed feedback; trace logs → SFT / decision / preference datasets |
| **Memory** | episodic memory with retrieval for in-context improvement; expert iteration from traced successes for in-weights improvement |
| **GNN** | self-supervised pretraining (GraphMAE `masked_feature`, GRACE `graph_contrastive`) → node classification fine-tuning |
| **Classic RL** | PPO with GAE on Gymnasium-style environments (built-in CartPole, `gymnasium` adapter) |
| **Serving** | in-process torch, Hugging Face, **SGLang**, **vLLM**, any OpenAI-compatible server; `toolbox serve` and `toolbox bench` |
| **Scale-out** | single device, DDP, FSDP; bf16/fp16 autocast; `torchrun`; SLURM with auto-requeue and resume |
| **Observability** | console, JSONL, TensorBoard, W&B; throughput, memory monitor, `torch.profiler` traces; rollout sample logs |
| **Data** | JSONL files, packaged offline samples (`builtin://`), Hugging Face Hub (`hf://org/name?split=train`) |

Pretraining is included only as a minimal teaching example (`experiment=lm_pretrain_tiny`). For
large-scale pretraining, torchtitan, Megatron-LM and nanotron are the right tools. Their
checkpoints load here through Hugging Face.

## Packaged experiments

| Experiment | What it does | Runs on |
| --- | --- | --- |
| `sft_tiny`, `dpo_tiny`, `reward_model_tiny`, `grpo_tiny` | the RLHF stages on a bundled arithmetic task | CPU, seconds |
| `decision_tiny`, `decision_rl_tiny` | decision model trained by CE / policy gradient | CPU |
| `sft_from_traces` | expert iteration on traced rollouts | CPU |
| `ppo_cartpole` | PPO actor-critic on CartPole | CPU |
| `gnn_pretrain` → `gnn_finetune` | GNN pretraining then node classification | CPU |
| `sft_qwen`, `dpo_qwen`, `grpo_qwen_gsm8k` | the same recipes on Qwen2.5-0.5B (+LoRA) | 1+ GPU |

The tiny experiments double as the integration test suite: every one of them runs end-to-end
in CI.

## Recipes

| Recipe | What it shows |
| --- | --- |
| [`recipes/rlhf_pipeline.sh`](recipes/rlhf_pipeline.sh) | SFT → reward model → DPO and GRPO (verifiable and learned reward), chained with `model.init_from` |
| [`recipes/self_improve.py`](recipes/self_improve.py) | expert iteration: GRPO rounds alternating with SFT on the policy's own traced successes |
| [`recipes/agent_tracing.py`](recipes/agent_tracing.py) | log an agent's decisions, attach delayed rewards, train a decision model on the trace |
| [`recipes/custom_objective.py`](recipes/custom_objective.py) | add a new loss without touching toolbox code |

## Examples

Techniques built on top of toolbox, one self-contained folder each. The index is
[`examples/README.md`](examples/README.md). The first is
[GMV](examples/gmv/): serving GNN outputs as views.

## How it fits together

```
toolbox CLI ─► Hydra config ─► ExperimentBuilder ─► registries (models, data, objectives, rewards, ...)
                                      │
                                      ▼
                       Trainer (template method: setup → step → eval → save)
                         │ emits events                         │ uses
                         ▼                                      ▼
   callbacks: loggers · checkpoints · tracing · memory     strategy: single · DDP · FSDP
```

```
src/toolbox/
├── core/         registry & plugins, events, retry, run context, decorators, constants
├── data/         tokenizers, repositories (local / HF Hub), datasets, collators
├── nn/           layers (attention, RoPE, RMSNorm, SwiGLU, graph convs) and models
├── objectives/   supervised · preference · policy · graph losses
├── rewards/      rl/      decision/      tracing/      memory/
├── trainer/      base loop, trainers, callbacks, optimizers, checkpoints
├── distributed/  serving/  launch/  eval/
├── configs/      the Hydra config tree
├── experiment.py ExperimentBuilder
└── cli.py        `toolbox train | launch | serve | bench | traces | list | config | env`
```

## Adding a technique

```python
from toolbox.core.registry import OBJECTIVES
from toolbox.objectives.base import PreferenceObjective

@OBJECTIVES.register("my_preference_loss")
class MyLoss(PreferenceObjective):
    def __call__(self, policy_chosen, policy_rejected, ref_chosen, ref_rejected): ...
```

Then select it with `objective=my_preference_loss`. To share it across projects, ship it in your
own package under the `toolbox.plugins` entry point. See
[Extending](https://lamng3.github.io/toolbox/guides/extending/).

## Development

```bash
uv pip install -e ".[dev,docs]"
make check        # ruff + mypy + pytest + mkdocs build --strict (same as CI)
make docs-serve   # live docs at http://127.0.0.1:8000
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](https://lamng3.github.io/toolbox/roadmap/).
Rollout engines with weight sync, more preference objectives, multi-turn agent RL and KV-cache
inference exercises come next.

## License

[CC BY 4.0](LICENSE). Copyright (c) 2026 Lam Nguyen.
