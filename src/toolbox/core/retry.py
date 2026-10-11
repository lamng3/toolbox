"""Retry with exponential backoff (pattern: *Retry Logic Refactor* + *Decorator*).

Network-bound operations — downloading from the Hugging Face Hub, submitting SLURM jobs,
polling an inference server — fail transiently. Instead of hand-rolled ``for``/``try`` loops at
each call site, describe the behaviour once with a :class:`RetryPolicy` and apply it with
:func:`retry` or :func:`call_with_retry`.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to retry and how long to wait in between.

    The delay before attempt ``n`` (1-based, counting retries) is
    ``min(initial_delay * backoff ** (n - 1), max_delay)`` plus up to ``jitter`` fraction of
    random noise, which avoids synchronised retries from many workers ("thundering herd").
    """

    max_attempts: int = 3
    initial_delay: float = 1.0
    backoff: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.1
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    def delay(self, retry_number: int) -> float:
        base = min(self.initial_delay * self.backoff ** (retry_number - 1), self.max_delay)
        return base * (1.0 + random.uniform(0.0, self.jitter))


NO_RETRY = RetryPolicy(max_attempts=1)
NETWORK_RETRY = RetryPolicy(max_attempts=5, initial_delay=1.0, backoff=2.0, max_delay=30.0)


def call_with_retry(
    fn: Callable[..., R],
    *args: Any,
    policy: RetryPolicy = NETWORK_RETRY,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> R:
    """Call ``fn(*args, **kwargs)``, retrying according to ``policy``.

    ``sleep`` is injectable so tests run instantly.
    """
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except policy.retry_on as exc:
            if attempt == policy.max_attempts:
                raise
            wait = policy.delay(attempt)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.2fs",
                getattr(fn, "__qualname__", fn),
                attempt,
                policy.max_attempts,
                exc,
                wait,
            )
            sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


def retry(
    policy: RetryPolicy = NETWORK_RETRY, *, sleep: Callable[[float], None] = time.sleep
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator form of :func:`call_with_retry`."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return call_with_retry(fn, *args, policy=policy, sleep=sleep, **kwargs)

        return wrapper

    return decorator
