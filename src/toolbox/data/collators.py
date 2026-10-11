"""Batch collation: turn a list of variable-length examples into padded tensors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from toolbox.core.constants import IGNORE_INDEX
from toolbox.core.types import Batch

INPUT_IDS_SUFFIX = "input_ids"
LABELS_SUFFIX = "labels"
MASK_SUFFIX = "attention_mask"


def pad_sequences(
    sequences: Sequence[Sequence[int]], pad_value: int, side: str = "right"
) -> torch.Tensor:
    """Pad integer sequences to a ``(B, max_len)`` long tensor on the ``left`` or ``right``."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    max_len = max((len(s) for s in sequences), default=0)
    out = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        values = torch.as_tensor(seq, dtype=torch.long)
        if side == "right":
            out[i, : len(seq)] = values
        else:
            out[i, max_len - len(seq) :] = values
    return out


class PaddingCollator:
    """Generic collator driven by field-name conventions.

    * ``*input_ids`` fields are padded with ``pad_token_id`` and get a matching
      ``*attention_mask`` field (``chosen_input_ids`` → ``chosen_attention_mask``).
    * ``*labels`` fields are padded with ``IGNORE_INDEX``.
    * Tensors are stacked; anything else (strings, dicts) is gathered into a list.

    One collator therefore serves LM pretraining, SFT and preference pairs.
    """

    def __init__(self, pad_token_id: int, padding_side: str = "right") -> None:
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, examples: list[dict[str, Any]]) -> Batch:
        batch: Batch = {}
        for key in examples[0]:
            values = [ex[key] for ex in examples]
            if key.endswith(INPUT_IDS_SUFFIX):
                ids = pad_sequences(values, self.pad_token_id, self.padding_side)
                batch[key] = ids
                lengths = torch.tensor([len(v) for v in values])
                batch[key.removesuffix(INPUT_IDS_SUFFIX) + MASK_SUFFIX] = _length_mask(
                    lengths, ids.shape[1], self.padding_side
                )
            elif key.endswith(LABELS_SUFFIX) and isinstance(values[0], list | tuple):
                batch[key] = pad_sequences(values, IGNORE_INDEX, self.padding_side)
            elif isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            elif isinstance(values[0], int | float):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        return batch


def _length_mask(lengths: torch.Tensor, max_len: int, side: str) -> torch.Tensor:
    positions = torch.arange(max_len)[None, :]
    if side == "right":
        return (positions < lengths[:, None]).long()
    return (positions >= (max_len - lengths)[:, None]).long()


def identity_collate(examples: list[Any]) -> Any:
    """For datasets whose single item is already a full batch (e.g. a whole graph)."""
    return examples[0]


def list_collate(examples: list[dict[str, Any]]) -> Batch:
    """Gather each field into a Python list (prompt/answer datasets for RL rollouts)."""
    return {key: [ex[key] for ex in examples] for key in examples[0]}
