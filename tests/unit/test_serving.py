from __future__ import annotations

import sys
import threading
import types
from collections.abc import Iterator

import pytest

from toolbox.data.tokenization import ByteTokenizer
from toolbox.nn.models import TinyGPT
from toolbox.serving import (
    OpenAICompatibleEngine,
    SamplingParams,
    ServerLaunchSpec,
    SGLangEngine,
    TorchEngine,
    TracedEngine,
    VLLMEngine,
    run_benchmark,
    serve_engine,
)
from toolbox.tracing import InMemoryTraceStore, Tracer

PARAMS = SamplingParams(max_new_tokens=4, temperature=0.0)


@pytest.fixture
def engine() -> TorchEngine:
    tok = ByteTokenizer()
    model = TinyGPT(tok.vocab_size, d_model=32, n_layers=1, n_heads=4, max_seq_len=64)
    return TorchEngine(model, tok, device="cpu")


def test_torch_engine_generates(engine: TorchEngine) -> None:
    out = engine.generate(["a", "bb"], SamplingParams(max_new_tokens=4, temperature=0.0, n=2))
    assert len(out) == 4 and all(c.num_tokens is not None for c in out)


def test_http_roundtrip(engine: TorchEngine) -> None:
    server = serve_engine(engine, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenAICompatibleEngine(f"http://127.0.0.1:{server.server_address[1]}/v1")
        assert client.health()
        assert len(client.generate(["hi", "yo"], PARAMS)) == 2
    finally:
        server.shutdown()


def test_traced_engine_logs_generations(engine: TorchEngine) -> None:
    store = InMemoryTraceStore()
    TracedEngine(engine, Tracer(store)).generate(["x"], PARAMS)
    (record,) = store.records()
    assert record.kind == "generation" and record.state == "x"


def test_benchmark(engine: TorchEngine) -> None:
    result = run_benchmark(engine, ["a"] * 5, PARAMS, batch_size=2, warmup_batches=0)
    assert result.num_requests == 5 and {"p50", "p99"} <= set(result.latency_ms)


@pytest.fixture
def fake_backends(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    sglang = types.ModuleType("sglang")

    class Engine:
        def __init__(self, model_path: str, **kw: object) -> None:
            self.model_path = model_path

        def generate(self, prompts: list[str], sampling: dict[str, object]) -> list[dict]:
            return [
                {
                    "text": f"<{p}>",
                    "meta_info": {"completion_tokens": 3, "finish_reason": {"type": "stop"}},
                }
                for p in prompts
            ]

        def shutdown(self) -> None: ...

    sglang.Engine = Engine  # type: ignore[attr-defined]
    vllm = types.ModuleType("vllm")

    class Out:
        def __init__(self, text: str) -> None:
            self.text, self.token_ids, self.finish_reason = text, [1, 2], "length"

    class Req:
        def __init__(self, prompt: str) -> None:
            self.prompt, self.outputs = prompt, [Out(f"<{prompt}>")]

    class LLM:
        def __init__(self, model: str, **kw: object) -> None: ...

        def generate(self, prompts: list[str], sampling: object) -> list[Req]:
            return [Req(p) for p in prompts]

    vllm.LLM = LLM  # type: ignore[attr-defined]
    vllm.SamplingParams = lambda **kw: kw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    yield


@pytest.mark.usefixtures("fake_backends")
def test_sglang_and_vllm_adapters_translate_outputs() -> None:
    for backend in (SGLangEngine("m"), VLLMEngine("m")):
        (c,) = backend.generate(["p"], PARAMS)
        assert c.text == "<p>" and c.prompt == "p" and c.num_tokens in (2, 3)


def test_server_launch_specs() -> None:
    assert "sglang.launch_server" in ServerLaunchSpec("sglang", "Qwen/x").shell()
    assert ServerLaunchSpec("vllm", "Qwen/x", port=8000).command()[:3] == [
        "vllm",
        "serve",
        "Qwen/x",
    ]
    with pytest.raises(ValueError):
        ServerLaunchSpec("tgi", "x").command()
