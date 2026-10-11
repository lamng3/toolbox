"""Process-wide run context and logging (pattern: *Singleton*).

There is exactly one run per process, and many modules need to know about it (output
directory, rank, run name). :class:`RunContext` is a thread-safe singleton holding that state.
It is *resettable* so tests stay isolated — the main pitfall of singletons is hidden global
state leaking between tests.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from toolbox.core.constants import ENV_LOCAL_RANK, ENV_RANK, ENV_WORLD_SIZE

LOGGER_NAME = "toolbox"


@dataclass
class RunContext:
    """Information about the current run, shared across the process."""

    run_name: str = "default"
    output_dir: Path = field(default_factory=lambda: Path("runs/default"))
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    _instance: ClassVar[RunContext | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def instance(cls) -> RunContext:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        rank=int(os.environ.get(ENV_RANK, 0)),
                        local_rank=int(os.environ.get(ENV_LOCAL_RANK, 0)),
                        world_size=int(os.environ.get(ENV_WORLD_SIZE, 1)),
                    )
        return cls._instance

    @classmethod
    def configure(cls, run_name: str, output_dir: str | Path) -> RunContext:
        ctx = cls.instance()
        ctx.run_name = run_name
        ctx.output_dir = Path(output_dir)
        return ctx

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


_logging_configured = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the ``toolbox`` logger, configuring a Rich handler on first use."""
    global _logging_configured
    if not _logging_configured:
        _configure_root_logger()
        _logging_configured = True
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def _configure_root_logger() -> None:
    root = logging.getLogger(LOGGER_NAME)
    if root.handlers:
        return
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        fmt = "%(message)s"
    except ImportError:  # pragma: no cover - rich is a core dependency
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(os.environ.get("TOOLBOX_LOG_LEVEL", "INFO"))
    root.propagate = False
