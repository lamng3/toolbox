"""Inference-engine adapters: torch (in-process), Hugging Face, SGLang, vLLM, OpenAI-HTTP."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from toolbox.core.constants import IGNORE_INDEX
from toolbox.core.decorators import require
from toolbox.core.env import resolve_device
from toolbox.core.registry import ENGINES, MODELS, TOKENIZERS
from toolbox.core.retry import NETWORK_RETRY, RetryPolicy, call_with_retry
from toolbox.data.collators import pad_sequences
from toolbox.serving.base import Completion, InferenceEngine, SamplingParams
from toolbox.tracing.records import ActionKind
from toolbox.tracing.tracer import Tracer

MS_PER_S = 1000.0
HTTP_TIMEOUT_S = 120.0


def _truncate_at_stop(text: str, stop: Sequence[str]) -> tuple[str, str | None]:
    cut = min((i for s in stop if (i := text.find(s)) >= 0), default=-1)
    return (text[:cut], "stop") if cut >= 0 else (text, None)


@ENGINES.register("torch")
class TorchEngine(InferenceEngine):
    """Runs any toolbox :class:`~toolbox.nn.models.base.CausalLM` in-process.

    Good for tests, tiny models and as the readable baseline when studying what SGLang/vLLM
    optimise (KV caching, continuous batching, paged attention, ...).
    """

    name = "torch"

    def __init__(
        self, model: Any, tokenizer: Any, checkpoint: str | None = None, device: str = "auto"
    ) -> None:
        self.tokenizer = (
            TOKENIZERS.build(tokenizer) if isinstance(tokenizer, str | Mapping) else tokenizer
        )
        if isinstance(model, Mapping):
            spec = {k: v for k, v in model.items() if k != "init_from"}
            model = MODELS.build(spec, inject={"vocab_size": self.tokenizer.vocab_size})
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state.get("model", state))
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], params: SamplingParams) -> list[Completion]:
        tok = self.tokenizer
        start = time.perf_counter()
        expanded = [p for p in prompts for _ in range(params.n)]
        ids = pad_sequences(
            [tok.encode(p, add_bos=True) for p in expanded], tok.pad_token_id, side="left"
        ).to(self.device)
        out = self.model.generate(
            ids,
            params.max_new_tokens,
            params.temperature,
            params.top_k,
            tok.eos_token_id,
            tok.pad_token_id,
        )
        latency = (time.perf_counter() - start) * MS_PER_S
        completions = []
        for prompt, row in zip(expanded, out.tolist()):
            row = [t for t in row if t not in (tok.pad_token_id, IGNORE_INDEX)]
            finish = "length"
            if tok.eos_token_id in row:
                row, finish = row[: row.index(tok.eos_token_id)], "stop"
            text, stopped = _truncate_at_stop(tok.decode(row), params.stop)
            completions.append(Completion(text, prompt, len(row), stopped or finish, latency))
        return completions


@ENGINES.register("hf")
def hf_engine(
    model_name_or_path: str, dtype: str = "bfloat16", device: str = "auto"
) -> TorchEngine:
    """Hugging Face ``transformers`` model behind the same interface as :class:`TorchEngine`."""
    from toolbox.data.tokenization import HFTokenizer
    from toolbox.nn.models.hf import HFCausalLM

    return TorchEngine(
        HFCausalLM(model_name_or_path, dtype=dtype), HFTokenizer(model_name_or_path), device=device
    )


@ENGINES.register("sglang")
class SGLangEngine(InferenceEngine):
    """In-process SGLang ``Engine`` (RadixAttention prefix caching, continuous batching).

    Install with ``pip install 'toolbox[serve-sglang]'`` on a CUDA machine. ``engine_kwargs``
    pass straight through (``tp_size``, ``mem_fraction_static``, ``dtype`` ...).
    """

    name = "sglang"

    def __init__(self, model_path: str, **engine_kwargs: Any) -> None:
        sglang = require("sglang", extra="serve-sglang")
        self.engine = sglang.Engine(model_path=model_path, **engine_kwargs)

    def generate(self, prompts: Sequence[str], params: SamplingParams) -> list[Completion]:
        sampling: dict[str, Any] = {
            "max_new_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "n": params.n,
        }
        if params.top_k is not None:
            sampling["top_k"] = params.top_k
        if params.stop:
            sampling["stop"] = list(params.stop)
        start = time.perf_counter()
        outputs = self.engine.generate(list(prompts), sampling)
        latency = (time.perf_counter() - start) * MS_PER_S
        expanded = [p for p in prompts for _ in range(params.n)]
        results = []
        for prompt, out in zip(expanded, outputs):
            meta = out.get("meta_info", {})
            finish = meta.get("finish_reason")
            results.append(
                Completion(
                    out["text"],
                    prompt,
                    meta.get("completion_tokens"),
                    finish.get("type") if isinstance(finish, dict) else finish,
                    latency,
                    meta,
                )
            )
        return results

    def close(self) -> None:
        self.engine.shutdown()


@ENGINES.register("vllm")
class VLLMEngine(InferenceEngine):
    """In-process vLLM ``LLM`` (PagedAttention, continuous batching).

    Install with ``pip install 'toolbox[serve-vllm]'``. ``engine_kwargs`` pass through
    (``tensor_parallel_size``, ``gpu_memory_utilization``, ``max_model_len`` ...).
    """

    name = "vllm"

    def __init__(self, model: str, **engine_kwargs: Any) -> None:
        self._vllm = require("vllm", extra="serve-vllm")
        self.llm = self._vllm.LLM(model=model, **engine_kwargs)

    def generate(self, prompts: Sequence[str], params: SamplingParams) -> list[Completion]:
        sampling = self._vllm.SamplingParams(
            n=params.n,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k if params.top_k is not None else -1,
            max_tokens=params.max_new_tokens,
            stop=list(params.stop) or None,
        )
        start = time.perf_counter()
        outputs = self.llm.generate(list(prompts), sampling)
        latency = (time.perf_counter() - start) * MS_PER_S
        return [
            Completion(o.text, req.prompt, len(o.token_ids), o.finish_reason, latency)
            for req in outputs
            for o in req.outputs
        ]


@ENGINES.register("openai_http")
class OpenAICompatibleEngine(InferenceEngine):
    """Client for any OpenAI-compatible ``/v1/completions`` server.

    Works with ``python -m sglang.launch_server``, ``vllm serve``, ``toolbox serve`` and hosted
    APIs. Requests are retried with exponential backoff.
    """

    name = "openai_http"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000/v1",
        model: str = "default",
        api_key: str | None = None,
        timeout: float = HTTP_TIMEOUT_S,
        retry: RetryPolicy = NETWORK_RETRY,
    ) -> None:
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.timeout, self.retry = timeout, retry

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode(), headers=headers
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result: dict[str, Any] = json.loads(response.read())
            return result

    def generate(self, prompts: Sequence[str], params: SamplingParams) -> list[Completion]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": list(prompts),
            "max_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "n": params.n,
        }
        if params.stop:
            payload["stop"] = list(params.stop)
        start = time.perf_counter()
        body = call_with_retry(self._post, "/completions", payload, policy=self.retry)
        latency = (time.perf_counter() - start) * MS_PER_S
        choices = sorted(body["choices"], key=lambda c: c.get("index", 0))
        expanded = [p for p in prompts for _ in range(params.n)]
        return [
            Completion(c["text"], p, None, c.get("finish_reason"), latency)
            for p, c in zip(expanded, choices)
        ]

    def health(self) -> bool:
        root = self.base_url.removesuffix("/v1")
        try:
            with urllib.request.urlopen(f"{root}/health", timeout=self.timeout):
                return True
        except (urllib.error.URLError, OSError):
            return False


class TracedEngine(InferenceEngine):
    """Decorator: logs every prompt → completion as a ``generation`` action."""

    def __init__(self, engine: InferenceEngine, tracer: Tracer, agent: str = "llm") -> None:
        self.engine, self.tracer, self.agent = engine, tracer, agent
        self.name = f"traced:{engine.name}"

    def generate(self, prompts: Sequence[str], params: SamplingParams) -> list[Completion]:
        completions = self.engine.generate(prompts, params)
        for c in completions:
            self.tracer.record(
                ActionKind.GENERATION,
                agent=self.agent,
                state=c.prompt,
                action=c.text,
                latency_ms=c.latency_ms,
                tokens=c.num_tokens,
                model=self.engine.name,
                metadata={"temperature": params.temperature, "finish_reason": c.finish_reason},
            )
        return completions

    def health(self) -> bool:
        return self.engine.health()

    def close(self) -> None:
        self.engine.close()
