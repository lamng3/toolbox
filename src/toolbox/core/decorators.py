"""Cross-cutting decorators (pattern: *Decorator*).

Each decorator adds one behaviour (timing, rank filtering, lazy optional imports) without the
wrapped function knowing about it.
"""

from __future__ import annotations

import functools
import importlib
import time
from collections.abc import Callable
from types import ModuleType
from typing import ParamSpec, TypeVar

from toolbox.core.logging import RunContext, get_logger

P = ParamSpec("P")
R = TypeVar("R")


def rank_zero_only(fn: Callable[P, R]) -> Callable[P, R | None]:
    """Run ``fn`` only on the global rank-0 process (logging, saving, printing)."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        if RunContext.instance().is_main_process:
            return fn(*args, **kwargs)
        return None

    return wrapper


def timed(fn: Callable[P, R]) -> Callable[P, R]:
    """Log the wall-clock duration of each call at DEBUG level."""
    log = get_logger("timing")

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.debug("%s took %.3fs", fn.__qualname__, time.perf_counter() - start)

    return wrapper


class MissingDependencyError(ImportError):
    """Raised when an optional backend is used without its extra installed."""


def require(module: str, extra: str) -> ModuleType:
    """Import an optional dependency or fail with an actionable message.

    Example:
        >>> vllm = require("vllm", extra="serve-vllm")  # doctest: +SKIP
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"'{module}' is required for this feature. Install it with: "
            f"pip install 'toolbox[{extra}]'  (or: uv pip install -e '.[{extra}]')"
        ) from exc
