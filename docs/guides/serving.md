# Serving and inference

Everything that generates text goes through one interface:

```python
class InferenceEngine:
    def generate(self, prompts: list[str], params: SamplingParams) -> list[Completion]: ...
```

| Engine | Backend | Notes |
| --- | --- | --- |
| `torch` | any toolbox `CausalLM`, in-process | readable reference: no KV cache, no batching tricks |
| `hf` | transformers `generate` | |
| `sglang` | `sglang.Engine` | RadixAttention prefix caching, continuous batching (`.[serve-sglang]`) |
| `vllm` | `vllm.LLM` | PagedAttention, continuous batching (`.[serve-vllm]`) |
| `openai_http` | any `/v1/completions` server | SGLang / vLLM servers, `toolbox serve`, hosted APIs; retries with backoff |

## Serve a model

```bash
toolbox serve --engine sglang --model Qwen/Qwen2.5-0.5B-Instruct          # python -m sglang.launch_server ...
toolbox serve --engine vllm   --model Qwen/Qwen2.5-0.5B-Instruct          # vllm serve ...
toolbox serve --engine torch  --checkpoint runs/sft_tiny/checkpoints/last.pt   # tiny OpenAI-style server
```

```bash
curl -s localhost:30000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "Q: 2+3=? A:", "max_tokens": 4, "temperature": 0}'
```

## Benchmark

```bash
toolbox bench --engine torch --num-prompts 64 --batch-size 16
toolbox bench --engine openai_http --base-url http://127.0.0.1:30000/v1
```

This reports requests/s, output tokens/s, and per-batch latency mean, p50, p90 and p99. The
prompts and sampling settings are identical across engines, so the comparison is fair.

## Studying inference optimisation

The `torch` engine is the baseline on purpose. Its `CausalLM.generate` recomputes the whole
prefix at every step. Good exercises:

1. Add a KV cache to `CausalSelfAttention` and measure the speed-up with `toolbox bench`.
2. Implement continuous batching on top of it.
3. Compare against SGLang (prefix caching helps GRPO, where G samples share a prompt) and vLLM.

[mini-sglang](https://github.com/sgl-project/mini-sglang) is a compact reference for SGLang's
internals.
