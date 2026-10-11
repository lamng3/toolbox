from __future__ import annotations

from pathlib import Path

import pytest
import torch

from toolbox.core.constants import IGNORE_INDEX
from toolbox.data import repository
from toolbox.data.collators import PaddingCollator, pad_sequences
from toolbox.data.datasets import sft, synthetic_graph, text_lm, train_eval_split
from toolbox.data.repository import InMemoryRepository, JsonlRepository, load_records
from toolbox.data.tokenization import ByteTokenizer


def test_byte_tokenizer_roundtrip_with_specials() -> None:
    tok = ByteTokenizer()
    ids = tok.encode("héllo", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_token_id and ids[-1] == tok.eos_token_id
    assert tok.decode(ids) == "héllo"
    assert tok.vocab_size == 259


def test_pad_sequences_left_and_right() -> None:
    right = pad_sequences([[1, 2], [3]], pad_value=0)
    left = pad_sequences([[1, 2], [3]], pad_value=0, side="left")
    assert right.tolist() == [[1, 2], [3, 0]]
    assert left.tolist() == [[1, 2], [0, 3]]


def test_padding_collator_conventions() -> None:
    batch = PaddingCollator(pad_token_id=9)(
        [
            {"chosen_input_ids": [1, 2, 3], "chosen_labels": [1, 2, 3], "target": 0, "text": "a"},
            {"chosen_input_ids": [4], "chosen_labels": [4], "target": 1, "text": "b"},
        ]
    )
    assert batch["chosen_input_ids"].tolist() == [[1, 2, 3], [4, 9, 9]]
    assert batch["chosen_attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["chosen_labels"][1, 1] == IGNORE_INDEX
    assert batch["target"].tolist() == [0, 1]
    assert batch["text"] == ["a", "b"]


def test_sft_masks_prompt_tokens() -> None:
    bundle = sft(ByteTokenizer(), eval_fraction=0.0)
    example = bundle.train[0]
    prompt_len = example["labels"].count(IGNORE_INDEX)
    assert prompt_len > 0
    assert example["labels"][prompt_len:] == example["input_ids"][prompt_len:]


def test_text_lm_blocks_have_fixed_length() -> None:
    bundle = text_lm(ByteTokenizer(), block_size=32)
    assert all(len(ex["input_ids"]) == 32 for ex in bundle.train)


def test_split_is_deterministic_and_keeps_training_data() -> None:
    a = train_eval_split(range(10), 0.3, seed=1)
    assert a == train_eval_split(range(10), 0.3, seed=1)
    assert len(a[1]) == 3
    assert len(train_eval_split([1], 0.9, seed=0)[0]) == 1


def test_synthetic_graph_info_and_masks() -> None:
    bundle = synthetic_graph(num_nodes=50, num_classes=3, feature_dim=4)
    graph = bundle.train[0]
    assert bundle.info == {"in_dim": 4, "num_classes": 3}
    masks = graph["train_mask"].int() + graph["val_mask"].int() + graph["test_mask"].int()
    assert torch.equal(masks, torch.ones(50, dtype=torch.int))


def test_repositories(tmp_path: Path) -> None:
    for repo in (InMemoryRepository(), JsonlRepository(tmp_path)):
        repo.write("a", [{"x": 1}])
        repo.append("a", [{"x": 2}])
        assert repo.read("a") == [{"x": 1}, {"x": 2}]
        assert repo.keys() == ["a"]
        with pytest.raises(KeyError):
            repo.read("missing")


def test_load_records_builtin_and_hf_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    assert len(load_records("builtin://sft.jsonl")) > 0
    seen = {}

    def fake_read(self: repository.HFHubRepository, split: str) -> list[dict[str, str]]:
        seen.update(path=self.path, config=self.config_name, split=split)
        return [{"q": "x"}]

    monkeypatch.setattr(repository.HFHubRepository, "read", fake_read)
    assert load_records("hf://openai/gsm8k?config=main&split=test") == [{"q": "x"}]
    assert seen == {"path": "openai/gsm8k", "config": "main", "split": "test"}
