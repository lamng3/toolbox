"""Recording agent actions: spans, a decorator, feedback, and pluggable sinks.

Usage::

    tracer = Tracer(JsonlTraceStore("runs/agent/traces"))

    with tracer.span("decision", agent="router", state=query, options=opts) as rec:
        decision = reader.decide(query, question)
        rec.action, rec.probs = decision.answer, decision.probs

    tracer.feedback(rec.id, reward=1.0)     # when the outcome is known

    @tracer.traced("tool_call", agent="search")
    def search(query: str) -> list[str]: ...

Spans nest: a span opened inside another becomes its child (same ``trace_id``, ``parent_id``
set), so one agent episode is a tree of atomic actions. Each finished record is pushed to every
*sink* (pattern: *Observer*); the bundled sink persists records through a repository
(pattern: *Repository*).
"""

from __future__ import annotations

import contextvars
import functools
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from toolbox.data.repository import InMemoryRepository, JsonlRepository, RecordRepository
from toolbox.tracing.records import ActionKind, ActionRecord

P = ParamSpec("P")
R = TypeVar("R")

Sink = Callable[[ActionRecord], None]
DEFAULT_TRACE_KEY = "actions"
MS_PER_S = 1000.0
MAX_REPR_CHARS = 2000

_current: contextvars.ContextVar[ActionRecord | None] = contextvars.ContextVar(
    "toolbox_current_action", default=None
)


class TraceStore:
    """Persists action records via any :class:`RecordRepository`."""

    def __init__(self, repository: RecordRepository, key: str = DEFAULT_TRACE_KEY) -> None:
        self.repository, self.key = repository, key
        self._lock = threading.Lock()

    def __call__(self, record: ActionRecord) -> None:
        with self._lock:
            self.repository.append(self.key, [record.to_dict()])

    def records(self) -> list[ActionRecord]:
        if not self.repository.exists(self.key):
            return []
        return [ActionRecord.from_dict(r) for r in self.repository.read(self.key)]

    def latest_by_id(self) -> dict[str, ActionRecord]:
        """Collapse the append-only log: later entries (feedback) win."""
        merged: dict[str, ActionRecord] = {}
        for record in self.records():
            merged[record.id] = record
        return merged


def JsonlTraceStore(root: str, key: str = DEFAULT_TRACE_KEY) -> TraceStore:  # noqa: N802
    """Convenience factory: traces as ``<root>/<key>.jsonl``."""
    return TraceStore(JsonlRepository(root), key)


def InMemoryTraceStore(key: str = DEFAULT_TRACE_KEY) -> TraceStore:  # noqa: N802
    return TraceStore(InMemoryRepository(), key)


def _preview(value: Any) -> Any:
    """Keep JSON-friendly values; truncate long reprs of everything else."""
    if value is None or isinstance(value, str | int | float | bool | list | dict):
        return value
    text = repr(value)
    return text if len(text) <= MAX_REPR_CHARS else text[:MAX_REPR_CHARS] + "…"


class Tracer:
    """Creates action records and fans them out to sinks."""

    def __init__(self, *sinks: Sink, enabled: bool = True) -> None:
        self.sinks: list[Sink] = list(sinks)
        self.enabled = enabled
        self._feedback_lock = threading.Lock()

    def add_sink(self, sink: Sink) -> None:
        self.sinks.append(sink)

    def emit(self, record: ActionRecord) -> ActionRecord:
        if self.enabled:
            for sink in self.sinks:
                sink(record)
        return record

    def record(self, kind: ActionKind | str, **fields: Any) -> ActionRecord:
        """Log an already-completed action in one call."""
        parent = _current.get()
        if parent is not None:
            fields.setdefault("trace_id", parent.trace_id)
            fields.setdefault("parent_id", parent.id)
        return self.emit(ActionRecord(kind=kind, **fields))

    @contextmanager
    def span(self, kind: ActionKind | str, **fields: Any) -> Iterator[ActionRecord]:
        """Open an action; fill ``action``/``probs``/... inside the block. Latency and errors
        are recorded automatically and the record is emitted on exit."""
        parent = _current.get()
        if parent is not None:
            fields.setdefault("trace_id", parent.trace_id)
            fields.setdefault("parent_id", parent.id)
        record = ActionRecord(kind=kind, **fields)
        token = _current.set(record)
        start = time.perf_counter()
        try:
            yield record
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.latency_ms = (time.perf_counter() - start) * MS_PER_S
            _current.reset(token)
            self.emit(record)

    def traced(
        self, kind: ActionKind | str = ActionKind.TOOL_CALL, agent: str | None = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator: each call becomes an action with its arguments as ``state`` and return
        value as ``action`` (pattern: *Decorator*)."""

        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                state = {
                    "args": [_preview(a) for a in args],
                    "kwargs": {k: _preview(v) for k, v in kwargs.items()},
                }
                with self.span(kind, agent=agent or fn.__qualname__, state=state) as rec:
                    result = fn(*args, **kwargs)
                    rec.action = _preview(result)
                    return result

            return wrapper

        return decorator

    def feedback(
        self,
        action_id: str,
        reward: float | None = None,
        outcome: Any = None,
        store: TraceStore | None = None,
    ) -> ActionRecord | None:
        """Attach a delayed reward/outcome to an earlier action.

        The log is append-only: the updated record is re-emitted with the same ``id`` and
        :meth:`TraceStore.latest_by_id` resolves the latest version.
        """
        with self._feedback_lock:
            store = store or next((s for s in self.sinks if isinstance(s, TraceStore)), None)
            if store is None:
                raise ValueError("feedback needs a TraceStore sink to look the action up")
            record = store.latest_by_id().get(action_id)
            if record is None:
                return None
            if reward is not None:
                record.reward = reward
            if outcome is not None:
                record.outcome = outcome
            record.metadata["feedback_at"] = time.time()
            return self.emit(record)


def current_action() -> ActionRecord | None:
    """The innermost open span in this context (thread / asyncio task)."""
    return _current.get()


_global_tracer: Tracer | None = None
_global_lock = threading.Lock()


def get_tracer() -> Tracer:
    """Process-wide default tracer (disabled, no sinks) — configure with :func:`set_tracer`.

    Library code calls ``get_tracer()`` so tracing is opt-in and free when unused
    (pattern: *Singleton*, kept replaceable for tests).
    """
    global _global_tracer
    if _global_tracer is None:
        with _global_lock:
            if _global_tracer is None:
                _global_tracer = Tracer(enabled=False)
    return _global_tracer


def set_tracer(tracer: Tracer | None) -> None:
    global _global_tracer
    with _global_lock:
        _global_tracer = tracer
