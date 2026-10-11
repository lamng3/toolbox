# toolbox

**toolbox** is an extensible toolbox for machine learning research. Every model, layer, loss,
reward and training loop is a component you select by name, so trying an idea is a config
change and comparisons stay fair. Standard libraries (Hugging Face, TRL, PyTorch Geometric,
AnyJev, SGLang, vLLM) plug in as backends instead of being rewritten. The main focus is
post-training and RL for language models; it also covers decision models, agent tracing, GNNs
and serving.

```bash
toolbox train experiment=grpo_tiny                                    # laptop CPU, seconds
toolbox train experiment=grpo_qwen_gsm8k                              # one GPU
toolbox launch experiment=grpo_qwen_gsm8k launcher=slurm launcher.gpus_per_node=4   # cluster
```

<div class="grid cards" markdown>

- **Get started**

    ---

    Install, run a tiny experiment in seconds, and learn the override syntax.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

- **Do research with it**

    ---

    Baseline → change one thing → compare → add your own component → scale to a cluster →
    write it up.

    [:octicons-arrow-right-24: Research workflow](getting-started/research.md)

- **Post-training and RL**

    ---

    SFT, DPO / IPO / SLiC, reward models, GRPO / Dr. GRPO / RLOO / PPO with verifiable,
    learned or LLM-judge rewards, on the built-in trainers or TRL.

    [:octicons-arrow-right-24: Online RL guide](guides/grpo.md)

- **Decision models and agents**

    ---

    Calibrated option probabilities, debiased over option orders. Log agent actions, attach
    rewards later, train on the log.

    [:octicons-arrow-right-24: Decision models](guides/decision-models.md)

- **Swap any layer or model**

    ---

    Graph convs, heads, norms, attention and feed-forward blocks by name; PyG layers and any
    `nn.Module` too.

    [:octicons-arrow-right-24: Swapping layers](guides/layers.md)

- **Examples**

    ---

    Self-contained folders with code, configs, tests and results: recipes, GMV (graph model
    views) and Jev (training decision models).

    [:octicons-arrow-right-24: Examples](examples/index.md)

</div>

## What's included

| Area | Components |
| --- | --- |
| Post-training | SFT · DPO, IPO, SLiC · Bradley–Terry reward models · [TRL backend](guides/trl.md) · stages chained with `model.init_from` |
| Online RL | GRPO, PPO-clip, REINFORCE × group-relative, Dr. GRPO, RLOO advantages · [PPO with GAE](guides/env-rl.md) for environments |
| Rewards | exact match, format, length penalty, weighted sums, learned reward model, LLM judge |
| Models and layers | any Hugging Face causal LM (+ LoRA) · GCN / GraphSAGE / GAT and any PyG layer · any `nn.Module` · every layer [swappable by name](guides/layers.md) |
| Decision models | option probabilities from next-token logits, debiased over option orders · CE, distillation, policy-gradient training |
| Agents | [action tracing](guides/tracing.md) with delayed rewards · traces → SFT / decision / DPO data · [episodic memory and expert iteration](guides/memory.md) |
| GNN | GraphMAE / GRACE pretraining → node classification |
| Serving | torch, Hugging Face, SGLang, vLLM, OpenAI-compatible HTTP · `toolbox serve`, `toolbox bench` |
| Scale-out | DDP, FSDP, bf16 / fp16 · torchrun · SLURM with automatic requeue and resume |
| Tracking | console, JSONL, TensorBoard, W&B · throughput, memory, `torch.profiler` |
| Data | local JSONL · `builtin://` offline samples · `hf://org/name?split=train` |

Every selectable name is in the [component catalog](concepts/components.md), which is generated
from the code. Where a standard library exists, toolbox uses it as a backend; see
[Standard libraries first](concepts/backends.md).

!!! note "Scope"
    Pretraining is only a minimal teaching example. For large-scale pretraining use
    torchtitan, Megatron-LM or nanotron, and load the checkpoints here through Hugging Face.

## Design

The code is also a reference for software design. Strategy, Factory, Dependency Injection,
Observer, Plugin, Repository, Builder, Template Method, Command and the other classic patterns
each appear where they do real work, and the [design patterns](concepts/design-patterns.md)
page maps each one to its code.
