"""Episodic memory for agents and continual self-improvement.

Two ways a model "improves over time from memory":

1. **In-context** — :class:`MemoryStore` retrieves similar past experiences (state, action,
   reward) and :func:`augment_prompt` puts the best ones in the prompt. No weights change.
2. **In-weights** — experiences are written as action traces (:mod:`toolbox.tracing`) and
   periodically distilled back into the model (``data=traces mode=sft min_reward=1``), i.e.
   expert iteration / STaR. ``recipes/self_improve.py`` runs this loop end-to-end.

The store is a *Repository* over records; the *Embedder* is a *Strategy* — the default hashing
embedder needs no model, and a sentence-transformer or the policy's own hidden states can be
plugged in.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from toolbox.core.registry import CALLBACKS, MEMORIES
from toolbox.data.repository import InMemoryRepository, JsonlRepository, RecordRepository
from toolbox.trainer.callbacks import Callback

if TYPE_CHECKING:
    from toolbox.trainer.base import BaseTrainer

TOKEN_PATTERN = re.compile(r"\w+")
MEMORY_KEY = "memories"


class Embedder(ABC):
    dim: int

    @abstractmethod
    def __call__(self, texts: Sequence[str]) -> torch.Tensor:
        """``(N, dim)`` L2-normalised embeddings."""


class HashingEmbedder(Embedder):
    """Bag of word uni/bi-grams hashed into ``dim`` buckets — zero-dependency baseline."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim

    def __call__(self, texts: Sequence[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), self.dim)
        for i, text in enumerate(texts):
            words = TOKEN_PATTERN.findall(text.lower())
            for gram in words + [f"{a} {b}" for a, b in itertools.pairwise(words)]:
                out[i, self._bucket(gram)] += 1.0
        return F.normalize(out, dim=-1)


class SentenceTransformerEmbedder(Embedder):
    """Any `sentence-transformers <https://sbert.net>`_ model, e.g. ``all-MiniLM-L6-v2``
    (``pip install 'toolbox[embeddings]'``). The standard choice for semantic retrieval."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None
    ) -> None:
        from toolbox.core.decorators import require

        st = require("sentence_transformers", extra="embeddings")
        self.model = st.SentenceTransformer(model_name, device=device)
        dimension = getattr(self.model, "get_embedding_dimension", None)
        dimension = dimension or self.model.get_sentence_embedding_dimension
        self.dim = int(dimension())

    def __call__(self, texts: Sequence[str]) -> torch.Tensor:
        vectors = self.model.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True)
        return vectors.float().cpu()


@dataclass
class MemoryItem:
    state: str
    action: str
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "metadata": self.metadata,
        }


@MEMORIES.register("episodic")
class MemoryStore:
    """Append-only experience store with similarity search.

    Args:
        path: persist to ``<path>/memories.jsonl`` (``None`` = in-memory only).
        min_reward: ignore experiences below this reward when retrieving.
    """

    def __init__(
        self,
        path: str | None = None,
        embedder: Embedder | None = None,
        min_reward: float | None = None,
    ) -> None:
        self.repository: RecordRepository = JsonlRepository(path) if path else InMemoryRepository()
        self.embedder = embedder or HashingEmbedder()
        self.min_reward = min_reward
        self.items: list[MemoryItem] = []
        self._matrix: torch.Tensor | None = None
        if self.repository.exists(MEMORY_KEY):
            self.items = [MemoryItem(**r) for r in self.repository.read(MEMORY_KEY)]

    def __len__(self) -> int:
        return len(self.items)

    def add(self, items: Sequence[MemoryItem]) -> None:
        self.items.extend(items)
        self.repository.append(MEMORY_KEY, [i.to_dict() for i in items])
        self._matrix = None  # invalidate the embedding cache

    def search(self, query: str, k: int = 3) -> list[tuple[MemoryItem, float]]:
        candidates = [
            i
            for i in self.items
            if self.min_reward is None or (i.reward or -math.inf) >= self.min_reward
        ]
        if not candidates or k <= 0:
            return []
        matrix = self.embedder([c.state for c in candidates])
        scores = matrix @ self.embedder([query])[0]
        top = torch.topk(scores, min(k, len(candidates)))
        return [(candidates[i], float(s)) for s, i in zip(top.values, top.indices.tolist())]


def augment_prompt(
    prompt: str,
    memories: Sequence[tuple[MemoryItem, float]],
    header: str = "Relevant past experience:",
) -> str:
    """Prepend retrieved experiences as few-shot examples."""
    if not memories:
        return prompt
    shots = "\n".join(f"- {m.state.strip()} -> {m.action.strip()}" for m, _ in memories)
    return f"{header}\n{shots}\n\n{prompt}"


@CALLBACKS.register("memory_writer")
class MemoryWriter(Callback):
    """Stores RL rollouts with reward ≥ ``min_reward`` as episodic memories."""

    def __init__(self, path: str | None = None, min_reward: float = 1.0) -> None:
        self.path, self.min_reward = path, min_reward
        self.store: MemoryStore | None = None

    def on_fit_start(self, trainer: BaseTrainer) -> None:
        if trainer.is_main_process:
            self.store = MemoryStore(self.path or f"{trainer.config.output_dir}/memory")

    def on_rollout_end(self, trainer: BaseTrainer, rollout: Any, rewards: torch.Tensor) -> None:
        if self.store is None:
            return
        keep = [
            MemoryItem(p, c, r, {"step": trainer.step})
            for p, c, r in zip(rollout.prompts, rollout.completions, rewards.tolist())
            if r >= self.min_reward
        ]
        if keep:
            self.store.add(keep)


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "MemoryItem",
    "MemoryStore",
    "MemoryWriter",
    "SentenceTransformerEmbedder",
    "augment_prompt",
]
