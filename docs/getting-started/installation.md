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
| `llm` | transformers, datasets, peft, accelerate, trl | Hugging Face models/tokenizers, LoRA, `hf://` datasets, the TRL trainer |
| `serve-sglang` | sglang | `SGLangEngine`, `toolbox serve --engine sglang` |
| `serve-vllm` | vllm | `VLLMEngine`, `toolbox serve --engine vllm` |
| `gnn` | torch-geometric | PyG layers and models (`pyg_*`, `pyg_model`), `planetoid` datasets |
| `jev` | anyjev, transformers, peft | the [Jev decision-model pipeline](../examples/jev.md) |
| `embeddings` | sentence-transformers | `SentenceTransformerEmbedder` for memory |
| `rl` | gymnasium | `GymnasiumAdapter` environments |
| `tracking` | wandb, tensorboard | `wandb` / `tensorboard` callbacks |
| `docs` | mkdocs-material, mkdocstrings | building this site |

```bash
uv pip install -e ".[llm,tracking]"          # typical GPU box
uv pip install -e ".[llm,serve-vllm]"        # + inference with vLLM
```

## Troubleshooting

`ModuleNotFoundError: No module named 'toolbox'` on macOS
:   The checkout is probably inside an iCloud-synced folder (Desktop or Documents with
    "Desktop & Documents Folders" turned on). iCloud marks everything inside dot-directories
    such as `.venv` as *hidden*, and Python ≥ 3.11.9 / 3.12 deliberately skips hidden `.pth`
    files, so the editable install disappears. You can confirm it with
    `ls -lO .venv/lib/python*/site-packages/*.pth`, which shows `hidden`. The fix is to move
    the repository out of iCloud (for example to `~/code/toolbox`), run `chflags -R nohidden .`
    once, and recreate the venv. A `chflags nohidden` alone doesn't stick, because iCloud
    re-applies the flag. Keeping `.git` and multi-GB virtualenvs out of iCloud sync is a good
    idea anyway.

CUDA build of PyTorch
:   Install the wheel that matches your driver first, e.g.
    `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`, then the toolbox.
