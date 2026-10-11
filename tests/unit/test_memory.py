from __future__ import annotations

from pathlib import Path

from toolbox.memory import MemoryItem, MemoryStore, augment_prompt


def test_search_returns_most_similar(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.add(
        [
            MemoryItem("how do I sort a list in python", "use sorted()", 1.0),
            MemoryItem("what is the capital of france", "paris", 1.0),
        ]
    )
    (best, score), *_ = store.search("sort a python list", k=1)
    assert best.action == "use sorted()" and score > 0
    assert len(MemoryStore(str(tmp_path))) == 2  # persisted


def test_min_reward_filters_and_prompt_augmentation() -> None:
    store = MemoryStore(min_reward=0.5)
    store.add([MemoryItem("2+2", "5", 0.0), MemoryItem("2+3", "5", 1.0)])
    hits = store.search("2+2", k=5)
    assert [m.state for m, _ in hits] == ["2+3"]
    assert augment_prompt("Q", hits).startswith("Relevant past experience:")
    assert augment_prompt("Q", []) == "Q"


def test_sentence_transformer_embedder_plugs_into_memory(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The standard embedding library is a drop-in Embedder (faked here: no download)."""
    import sys
    import types

    import torch

    from toolbox.memory import SentenceTransformerEmbedder

    class FakeST:
        def __init__(self, name: str, device: str | None = None) -> None: ...

        def get_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts, convert_to_tensor, normalize_embeddings):  # type: ignore[no-untyped-def]
            return torch.tensor([[1.0, 0.0] if "sort" in t else [0.0, 1.0] for t in texts])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST)
    )
    store = MemoryStore(embedder=SentenceTransformerEmbedder("fake"))
    store.add([MemoryItem("sort numbers", "sorted()", 1.0), MemoryItem("capital", "paris", 1.0)])
    assert store.search("how to sort", k=1)[0][0].action == "sorted()"
