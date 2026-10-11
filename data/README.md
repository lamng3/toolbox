# data/

Local datasets live here and are **not** committed (see `.gitignore`).

| Folder | Purpose |
| --- | --- |
| `raw/` | Downloads as-is (e.g. Planetoid graphs land in `raw/planetoid`). |
| `processed/` | Your JSONL files, ready for a dataset factory. |

Point any dataset at a file, a packaged sample, or the Hugging Face Hub:

```bash
toolbox train experiment=sft_qwen data.path=data/processed/my_sft.jsonl
toolbox train experiment=sft_tiny data.path=builtin://sft.jsonl
toolbox train experiment=grpo_qwen_gsm8k data.path='hf://openai/gsm8k?config=main&split=train'
```

Record formats (one JSON object per line):

| Dataset | Fields |
| --- | --- |
| `sft` | `prompt`, `response` |
| `preference` | `prompt`, `chosen`, `rejected` |
| `prompts` (RL) | `prompt`, `answer` (reference for verifiable rewards) |
| `decisions` | `state`, `question`, `options`, `answer`, optional `probs` |
| `traces` | action records written by `toolbox.tracing` |
