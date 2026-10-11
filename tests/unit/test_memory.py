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
