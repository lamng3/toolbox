# Installation

toolbox needs Python ≥ 3.10 and PyTorch ≥ 2.2. [uv](https://docs.astral.sh/uv/) is the
recommended installer.

```bash
git clone https://github.com/lamng3/toolbox.git
cd toolbox
uv venv --python 3.11
uv pip install -e ".[dev]"          # core + test/lint tools
.venv/bin/toolbox env               # print versions, GPUs, optional backends
```

## Optional extras

Heavy backends are optional. They are imported lazily, and if one is missing you get a clear
error naming the extra to install.

| Extra | Installs | Needed for |
| --- | --- | --- |
| `llm` | transformers, datasets, peft, accelerate | Hugging Face models/tokenizers, LoRA, `hf://` datasets |
| `serve-sglang` | sglang | `SGLangEngine`, `toolbox serve --engine sglang` |
| `serve-vllm` | vllm | `VLLMEngine`, `toolbox serve --engine vllm` |
| `gnn` | torch-geometric | `planetoid` datasets (Cora, CiteSeer, PubMed) |
| `rl` | gymnasium | `GymnasiumAdapter` environments |
| `tracking` | wandb, tensorboard | `wandb` / `tensorboard` callbacks |
| `dist` | deepspeed | DeepSpeed integration |
| `docs` | mkdocs-material, mkdocstrings | building this site |

```bash
uv pip install -e ".[llm,tracking]"          # typical GPU box
uv pip install -e ".[llm,serve-vllm]"        # + inference with vLLM
```

## Troubleshooting

`ModuleNotFoundError: No module named 'toolbox'` on macOS
:   Some macOS setups mark files inside `.venv` as *hidden*, and Python 3.11.14+ skips
    hidden `.pth` files, which breaks editable installs. Run
    `chflags nohidden .venv/lib/python*/site-packages/*.pth`, or `export PYTHONPATH=src`
    (the Makefile does this for you).

CUDA build of PyTorch
:   Install the wheel that matches your driver first, e.g.
    `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`, then the toolbox.
