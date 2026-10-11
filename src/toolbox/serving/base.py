"""The inference-engine interface shared by every serving backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SamplingParams:
    """Backend-neutral sampling settings; adapters translate them to each engine's API."""

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    n: int = 1


@dataclass
class Completion:
    text: str
    prompt: str = ""
    num_tokens: int | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class InferenceEngine(ABC):
    """Generate text for a batch of prompts.

    Implementations are *adapters* over SGLang, vLLM, Hugging Face, an in-process toolbox model,
    or any OpenAI-compatible HTTP server. Rewards (LLM-as-judge), evaluation and the
    benchmark depend only on this interface (pattern: *Adapter* + *Polymorphism*).
    """

    name: str = "engine"

    @abstractmethod
    def generate(self, prompts: Sequence[str], params: SamplingParams) -> list[Completion]: ...

    def health(self) -> bool:
        return True

    def close(self) -> None:
        """Release GPU memory / connections."""

    def __enter__(self) -> InferenceEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
