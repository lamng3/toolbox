# Standard libraries first

toolbox is a place to **combine** the best tools, not to rewrite them. If a well-maintained
library already does something, toolbox uses it as a backend behind the same interfaces as
everything else. Custom code is written only when there's a reason: to teach how something
works, to change its internals, or because nothing standard exists.

That gives each area two layers:

* **Standard backend:** the library people use in production and papers. Use it by default.
* **Built-in reference:** a short, readable implementation for teaching and experimenting.
  Where both exist, the built-in is tested against the standard one.

| Area | Standard backend in toolbox | Built-in reference | Select with |
| --- | --- | --- | --- |
| Language models, tokenizers, LoRA | Hugging Face `transformers`, `peft` | `tiny_gpt`, byte tokenizer | `model=qwen2_5_0_5b_lora`, `tokenizer=qwen2_5` |
| Post-training (SFT, DPO, GRPO, reward models) | [TRL](../guides/trl.md) | `supervised`, `preference`, `grpo` trainers | `trainer.name=trl trainer.algorithm=grpo` |
| GNN layers and models | PyTorch Geometric | `gcn`, `sage`, `gat` (parity-tested against PyG) | `model.conv=pyg_gcn`, `model.name=pyg_model` |
| Your own model | any `nn.Module` | — | `model: {name: torch_module, target: pkg.mod:Class}` |
| Optimizers, LR schedules | `torch.optim`, `transformers` schedules | `adamw`, `cosine` ... | `optimizer: {name: torch, optimizer: RAdam}`, `scheduler: {name: hf, schedule: cosine}` |
| Datasets | Hugging Face `datasets` | JSONL, `builtin://` samples | `data.path=hf://org/name?split=train` |
| Decision models | [AnyJev](https://github.com/nokia-applied-research/AnyJev) (readout, evaluation) | `toolbox.decision` | [Jev pipeline](https://github.com/lamng3/toolbox/tree/main/examples/jev) |
| Serving | SGLang, vLLM | `torch` engine | `toolbox serve --engine sglang` |
| RL environments | Gymnasium | `cartpole` | `env=gymnasium env.env_id=...` |
| Embeddings for memory | sentence-transformers | hashing embedder | `SentenceTransformerEmbedder(...)` |
| Configs, tracking | Hydra, Weights & Biases, TensorBoard | console, JSONL | `callbacks` |

Each backend is an optional extra (`pip install 'toolbox[llm]'`, `[gnn]`, `[jev]`,
`[serve-vllm]`, `[embeddings]` ...), imported only when used. If an extra is missing, the error
names it.

## Planned backends

Areas where a standard exists but toolbox still uses its own code are tracked on the
[roadmap](../roadmap.md): Accelerate / DeepSpeed for multi-GPU, OpenTelemetry for agent
traces, and lm-evaluation-harness for evaluation.

## Adding a backend

A backend is an *adapter*: a registered component that translates between the library's API
and the toolbox interface for that role. Examples to copy:

* a layer: `src/toolbox/nn/layers/pyg.py`
* a model: `src/toolbox/nn/models/external.py`
* a trainer: `src/toolbox/trainer/trl_backend.py`
* an inference engine: `src/toolbox/serving/engines.py`
