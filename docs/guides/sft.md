# Supervised fine-tuning

SFT minimises next-token cross-entropy on the **response** tokens of prompt/response pairs.
Prompt tokens are labelled `IGNORE_INDEX` (-100), so the model is not trained to reproduce the
prompt.

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|}\sum_{t} \log \pi_\theta(y_t \mid x, y_{<t})
$$

## Data

JSONL with `prompt` and `response`:

```json
{"prompt": "Q: 3+4=? A:", "response": " 7"}
```

```bash
toolbox train experiment=sft_qwen data.path=data/processed/my_sft.jsonl data.chat_template=true
```

`chat_template=true` wraps the prompt with the tokenizer's chat template (Hugging Face
tokenizers that define one). The `sft` dataset factory also takes `prompt_key` /
`response_key` to match other schemas, and `path` accepts `hf://org/name?split=train`.

## Model choices

| Config | When |
| --- | --- |
| `model=tiny_gpt` | wiring checks on CPU |
| `model=qwen2_5_0_5b` | full fine-tuning, one GPU |
| `model=qwen2_5_0_5b_lora` | LoRA (r=16, all linear layers) to save memory |
| `model.model_name_or_path=<any HF id>` | anything `AutoModelForCausalLM` loads |

Useful knobs: `trainer.precision=bf16`, `trainer.grad_accum_steps=8`,
`model.gradient_checkpointing=true`, `strategy=fsdp`.

## From traces

`experiment=sft_from_traces` fine-tunes on logged agent or policy actions, optionally keeping
only rewarded ones (`data.min_reward=1`). See [Agent tracing](tracing.md) and
[Memory and self-improvement](memory.md).

!!! tip "Production runs"
    The same config runs on [TRL](trl.md) with `trainer.name=trl trainer.algorithm=sft`.
    TRL is the standard post-training library; the built-in trainer is the readable reference.
