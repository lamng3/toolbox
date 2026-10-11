# Post-training with TRL

[TRL](https://github.com/huggingface/trl) is the standard library for post-training Hugging Face
models. The `trl` trainer runs TRL's own `SFTTrainer`, `DPOTrainer`, `GRPOTrainer` or
`RewardTrainer`, and toolbox supplies everything around it:

| toolbox provides | TRL does |
| --- | --- |
| configs and command-line overrides | the training loop and optimisation |
| data sources: JSONL, `builtin://`, `hf://` | tokenisation, packing, generation |
| toolbox rewards (exact match, LLM judge ...) as GRPO reward functions | GRPO rollouts and advantages |
| LoRA settings from the model spec | PEFT integration |
| toolbox callbacks (console, JSONL, W&B ...) | logging hooks |
| launchers: torchrun, SLURM | distributed execution via Accelerate |

```yaml
model:   {name: hf_causal_lm, model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct, lora: {r: 16}}
data:    {name: prompts, path: "hf://openai/gsm8k?config=main&split=train",
          prompt_key: question, answer_key: answer}
reward:  {name: exact_match}
trainer:
  name: trl
  algorithm: grpo                 # sft | dpo | grpo | reward
  max_steps: 500
  batch_size: 8
  precision: bf16
  args: {num_generations: 8, max_completion_length: 256, beta: 0.04}   # any TRL config option
```

```bash
toolbox train experiment=grpo_qwen_gsm8k trainer.name=trl trainer.algorithm=grpo
```

`args` goes to the TRL config class unchanged (`SFTConfig`, `DPOConfig`, `GRPOConfig`,
`RewardConfig`), so every TRL option is available. Unknown options are rejected before training
starts.

| `algorithm` | Data records | TRL trainer |
| --- | --- | --- |
| `sft` | `prompt`, `response` | `SFTTrainer` (prompt–completion) |
| `dpo` | `prompt`, `chosen`, `rejected` | `DPOTrainer` |
| `reward` | `prompt`, `chosen`, `rejected` | `RewardTrainer` |
| `grpo` | `prompt`, `answer` (reference passed to the reward) | `GRPOTrainer` |

Set `data.chat_template: true` to send conversational records, so TRL applies the model's chat
template.

!!! warning "Hardware"
    TRL ≥ 1.0 computes log-probabilities for SFT, DPO and GRPO with a Triton kernel, which
    needs Linux with a GPU. Reward-model training also runs on CPU and macOS. On a laptop, use
    toolbox's built-in trainers (`supervised`, `preference`, `grpo`), which run anywhere.

## When to use which

| Use | When |
| --- | --- |
| `trainer.name=trl` | real post-training runs on GPUs; you want TRL's maintained, optimised implementations |
| `supervised` / `preference` / `grpo` | reading or modifying the algorithm itself, CPU experiments, new objectives that TRL doesn't have |

Both read the same configs, data and rewards, so you can prototype a new loss in the built-in
trainer and scale it on TRL.
