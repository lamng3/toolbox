"""The value every dataset factory returns."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from torch.utils.data import Dataset

T = TypeVar("T")


class ListDataset(Dataset[T]):
    """A map-style dataset over an in-memory list."""

    def __init__(self, items: Sequence[T]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]


@dataclass
class DataBundle:
    """Train/eval splits plus how to collate them.

    ``info`` carries data-derived facts the model needs (``in_dim``, ``num_classes``,
    ``obs_dim`` ...). :class:`~toolbox.experiment.ExperimentBuilder` injects them into the
    model constructor so configs do not have to repeat them.
    """

    train: Dataset[Any]
    eval: Dataset[Any] | None
    collate_fn: Callable[[list[Any]], Any]
    info: dict[str, Any] = field(default_factory=dict)


def train_eval_split(
    items: Sequence[T], eval_fraction: float, seed: int
) -> tuple[list[T], list[T]]:
    """Deterministic shuffle-and-split. Always keeps at least one training example."""
    items = list(items)
    random.Random(seed).shuffle(items)
    n_eval = min(int(len(items) * eval_fraction), len(items) - 1)
    return items[n_eval:], items[:n_eval]
