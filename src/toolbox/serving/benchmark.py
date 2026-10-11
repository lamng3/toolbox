"""Throughput/latency benchmark for any :class:`InferenceEngine`.

Use it to compare backends (torch vs. SGLang vs. vLLM) or to measure an inference
optimisation you are working on, with identical prompts and sampling settings.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from toolbox.serving.base import InferenceEngine, SamplingParams

PERCENTILES = (50, 90, 99)


@dataclass
class BenchmarkResult:
    engine: str
    num_requests: int
    batch_size: int
    total_time_s: float
    requests_per_s: float
    output_tokens_per_s: float | None
    latency_ms: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentile(values: Sequence[float], q: int) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1)))
    return ordered[index]


def run_benchmark(
    engine: InferenceEngine,
    prompts: Sequence[str],
    params: SamplingParams,
    batch_size: int = 8,
    warmup_batches: int = 1,
) -> BenchmarkResult:
    """Send ``prompts`` in batches; report per-batch latency percentiles and throughput."""
    batches = [list(prompts[i : i + batch_size]) for i in range(0, len(prompts), batch_size)]
    for batch in batches[:warmup_batches]:
        engine.generate(batch, params)
    latencies, tokens, counted = [], 0, True
    start = time.perf_counter()
    for batch in batches:
        t0 = time.perf_counter()
        completions = engine.generate(batch, params)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        for c in completions:
            if c.num_tokens is None:
                counted = False
            else:
                tokens += c.num_tokens
    total = time.perf_counter() - start
    return BenchmarkResult(
        engine=engine.name,
        num_requests=len(prompts),
        batch_size=batch_size,
        total_time_s=total,
        requests_per_s=len(prompts) / total if total else 0.0,
        output_tokens_per_s=tokens / total if counted and total else None,
        latency_ms={
            "mean": statistics.fmean(latencies),
            **{f"p{q}": _percentile(latencies, q) for q in PERCENTILES},
        },
    )
