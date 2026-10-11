"""A minimal synchronous event bus (pattern: *Observer*).

The trainer *publishes* lifecycle events; loggers, checkpointers, early stopping, memory
writers, profilers ... *subscribe*. The trainer never needs to know who is listening, so adding
a new side effect never means editing the training loop.
"""

from __future__ import annotations

import enum
from collections import defaultdict
from collections.abc import Callable
from typing import Any

Handler = Callable[..., None]


class Event(str, enum.Enum):
    """Lifecycle events emitted by :class:`toolbox.trainer.BaseTrainer`."""

    FIT_START = "fit_start"
    STEP_START = "step_start"
    STEP_END = "step_end"
    EVAL_END = "eval_end"
    CHECKPOINT_SAVED = "checkpoint_saved"
    ROLLOUT_END = "rollout_end"
    FIT_END = "fit_end"
    EXCEPTION = "exception"


class EventBus:
    """Publish/subscribe dispatcher. Handlers are called in subscription order."""

    def __init__(self) -> None:
        self._handlers: dict[Event, list[Handler]] = defaultdict(list)

    def subscribe(self, event: Event, handler: Handler) -> Callable[[], None]:
        """Register ``handler`` for ``event``; returns a function that unsubscribes it."""
        self._handlers[event].append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers[event]:
                self._handlers[event].remove(handler)

        return unsubscribe

    def emit(self, event: Event, **payload: Any) -> None:
        for handler in list(self._handlers[event]):
            handler(**payload)

    def handler_count(self, event: Event) -> int:
        return len(self._handlers[event])
