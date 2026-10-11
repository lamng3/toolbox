"""Storage abstraction for records and artifacts (pattern: *Repository*).

Datasets, traces, memories and checkpoints are read and written through a
:class:`RecordRepository`, so training code never hard-codes *where* data lives. Swap a local
directory for the Hugging Face Hub (or your own S3/GCS implementation) in config, not in code.
"""

from __future__ import annotations

import json
import urllib.parse
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from toolbox.core.decorators import require
from toolbox.core.retry import NETWORK_RETRY, call_with_retry

Record = dict[str, Any]

BUILTIN_DATA_DIR = Path(__file__).parent / "builtin"
BUILTIN_PREFIX = "builtin://"


def resolve_path(path: str | Path) -> Path:
    """Expand ``builtin://name`` to a file shipped with the package."""
    text = str(path)
    if text.startswith(BUILTIN_PREFIX):
        return BUILTIN_DATA_DIR / text.removeprefix(BUILTIN_PREFIX)
    return Path(text).expanduser()


class RecordRepository(ABC):
    """A keyed collection of record lists (one key ≈ one dataset split / one table)."""

    @abstractmethod
    def read(self, key: str) -> list[Record]: ...

    @abstractmethod
    def write(self, key: str, records: Iterable[Record]) -> None: ...

    @abstractmethod
    def append(self, key: str, records: Iterable[Record]) -> None: ...

    @abstractmethod
    def keys(self) -> list[str]: ...

    def exists(self, key: str) -> bool:
        return key in self.keys()

    def iter(self, key: str) -> Iterator[Record]:
        yield from self.read(key)


class InMemoryRepository(RecordRepository):
    """Dict-backed repository — the test double for every other implementation."""

    def __init__(self) -> None:
        self._data: dict[str, list[Record]] = {}

    def read(self, key: str) -> list[Record]:
        if key not in self._data:
            raise KeyError(key)
        return list(self._data[key])

    def write(self, key: str, records: Iterable[Record]) -> None:
        self._data[key] = list(records)

    def append(self, key: str, records: Iterable[Record]) -> None:
        self._data.setdefault(key, []).extend(records)

    def keys(self) -> list[str]:
        return sorted(self._data)


class JsonlRepository(RecordRepository):
    """One ``<key>.jsonl`` file per key under ``root`` (append-friendly, diff-friendly)."""

    SUFFIX = ".jsonl"

    def __init__(self, root: str | Path) -> None:
        self.root = resolve_path(root)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}{self.SUFFIX}"

    def read(self, key: str) -> list[Record]:
        return list(self.iter(key))

    def iter(self, key: str) -> Iterator[Record]:
        path = self._path(key)
        if not path.exists():
            raise KeyError(f"{path} does not exist")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def write(self, key: str, records: Iterable[Record]) -> None:
        self._dump(key, records, mode="w")

    def append(self, key: str, records: Iterable[Record]) -> None:
        self._dump(key, records, mode="a")

    def _dump(self, key: str, records: Iterable[Record], mode: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode, encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def keys(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob(f"*{self.SUFFIX}"))


class HFHubRepository(RecordRepository):
    """Read-only access to a Hugging Face dataset; ``key`` is the split name.

    Downloads are wrapped in :func:`~toolbox.core.retry.call_with_retry`, because the Hub
    occasionally rate-limits or times out on shared HPC login nodes.
    """

    def __init__(self, path: str, config_name: str | None = None) -> None:
        self.path, self.config_name = path, config_name

    def read(self, key: str) -> list[Record]:
        datasets = require("datasets", extra="llm")
        ds = call_with_retry(
            datasets.load_dataset, self.path, self.config_name, split=key, policy=NETWORK_RETRY
        )
        return [dict(row) for row in ds]

    def write(self, key: str, records: Iterable[Record]) -> None:
        raise NotImplementedError("HFHubRepository is read-only; push with `datasets` directly")

    def append(self, key: str, records: Iterable[Record]) -> None:
        self.write(key, records)

    def keys(self) -> list[str]:
        datasets = require("datasets", extra="llm")
        return list(datasets.get_dataset_split_names(self.path, self.config_name))


HF_PREFIX = "hf://"
DEFAULT_HF_SPLIT = "train"


def load_records(path: str | Path) -> list[Record]:
    """Load records from a URI — the one entry point every dataset factory uses.

    * ``builtin://sft.jsonl`` — tiny datasets shipped with the package.
    * ``data/my.jsonl`` — any local JSONL file.
    * ``hf://openai/gsm8k?config=main&split=train`` — a Hugging Face Hub dataset.
    """
    text = str(path)
    if text.startswith(HF_PREFIX):
        parsed = urllib.parse.urlparse(text)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        repo = f"{parsed.netloc}{parsed.path}"
        return HFHubRepository(repo, query.get("config")).read(query.get("split", DEFAULT_HF_SPLIT))
    resolved = resolve_path(path)
    return JsonlRepository(resolved.parent).read(resolved.stem)
