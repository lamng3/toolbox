# Swapping layers

Every building block inside a toolbox model is a registered **layer**. A model never constructs
a convolution, MLP, norm, activation, attention or feed-forward block directly. It asks the
registry for that *role* by name. So you can test a new architecture by changing a config value,
and a layer you write yourself plugs in exactly like a built-in.

## Layer roles

Each role has a fixed constructor contract. Anything registered under a role must accept those
arguments (plus any extra keyword arguments of its own) and implement the forward signature.

| Role (registry) | Constructor contract | Forward | Built-ins |
| --- | --- | --- | --- |
| `graph_conv` (`GRAPH_CONVS`) | `(in_dim, out_dim, **kw)` | `(x, edge_index)` | `gcn`, `sage`, `gat` |
| `mlp` (`MLPS`) | `(in_dim, out_dim, **kw)` | `(x)` | `linear`, `mlp` |
| `norm` (`NORMS`) | `(dim, **kw)` | `(x)` | `rmsnorm`, `layernorm`, `batchnorm`, `identity` |
| `activation` (`ACTIVATIONS`) | `(**kw)` | `(x)` | `relu`, `gelu`, `silu`, `tanh`, `identity` |
| `feedforward` (`FEEDFORWARDS`) | `(d_model, hidden_dim, **kw)` | `(x)` | `swiglu`, `gelu_mlp` |
| `attention` (`ATTENTIONS`) | `(d_model, n_heads, max_seq_len, n_kv_heads, dropout, **kw)` | `(x, attention_mask)` | `causal_self_attention` |

The current list is always in the [component catalog](../concepts/components.md), or in the
terminal with `toolbox list graph_conv`.

## Model slots

A slot takes a **layer spec**: a registered name, or `{name: ..., **kwargs}` for a layer with
options. The model fills in the contract arguments (dimensions) itself. Optional slots accept
`null`.

| Model | Slots |
| --- | --- |
| `gnn_node_classifier` | `conv`, `activation`, `norm`, `head` |
| `gnn_pretrainer` | `conv`, `activation`, `norm`, `decoder`, `projector` |
| `gnn_encoder` | `conv`, `activation`, `norm` |
| `tiny_gpt` | `attention`, `ffn`, `norm` (in every block and the final norm) |
| `tiny_reward_model` | `attention`, `ffn`, `norm`, `head` |
| `actor_critic` | `actor`, `critic`, `activation` |

## From config

```bash
toolbox train experiment=gnn_finetune model.conv=sage model.norm=layernorm
toolbox train experiment=gnn_finetune 'model.conv={name:gat,heads:8}' \
    'model.head={name:mlp,hidden_dims:[64],dropout:0.2}'
toolbox train experiment=lm_pretrain_tiny model.ffn=gelu_mlp model.norm=layernorm
```

or in YAML:

```yaml
model:
  name: tiny_gpt
  ffn: {name: gelu_mlp, bias: false}
  norm: layernorm
```

## Your own layer

Register it under its role, then select it by name. Nothing in toolbox needs to change.

```python
import torch
from torch import nn
from toolbox.core.registry import FEEDFORWARDS, MLPS

@FEEDFORWARDS.register("relu2")
class ReLUSquared(nn.Module):
    """Primer-style feed-forward: W2(relu(W1 x)^2)."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1, self.w2 = nn.Linear(d_model, hidden_dim), nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.relu(self.w1(x)) ** 2)


@MLPS.register("residual_head")
class ResidualHead(nn.Module):
    """A head with a residual hidden layer."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.inner = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, in_dim))
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x + self.inner(x))
```

```python
from toolbox.experiment import ExperimentBuilder, load_config

builder = ExperimentBuilder.from_config(load_config(["experiment=lm_pretrain_tiny"]))
builder.with_model("tiny_gpt", d_model=64, n_layers=2, ffn="relu2").build().fit()
```

[`recipes/custom_layer.py`](https://github.com/lamng3/toolbox/blob/main/recipes/custom_layer.py)
runs this end to end: a custom feed-forward in the transformer and a custom head on a GNN, each
compared with the built-in.

The CLI only sees layers whose module has been imported. Use them from a script, as above, or
ship them as a [plugin](extending.md#shipping-it-as-a-plugin), after which `model.ffn=relu2`
works from the command line too.

## Architecture ablations

A loop over specs is an ablation study:

```python
for ffn in ("swiglu", "gelu_mlp", "relu2"):
    for norm in ("rmsnorm", "layernorm"):
        trainer = (ExperimentBuilder(f"ablate-{ffn}-{norm}")
                   .with_tokenizer("byte").with_data("text_lm")
                   .with_model("tiny_gpt", ffn=ffn, norm=norm)
                   .with_objective("lm_ce").with_trainer("supervised", max_steps=200)
                   .build())
        print(ffn, norm, trainer.fit()["eval/loss"])
```

On a cluster, submit each one with `toolbox launch ... launcher=slurm`. See
[GPU VMs and SLURM](../getting-started/hpc.md#sweeps).
