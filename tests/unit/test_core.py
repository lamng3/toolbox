from __future__ import annotations

import pytest

from toolbox.core.events import Event, EventBus
from toolbox.core.logging import RunContext
from toolbox.core.retry import RetryPolicy, call_with_retry, retry


def test_retry_succeeds_after_transient_failures() -> None:
    calls, sleeps = [], []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("boom")
        return "ok"

    policy = RetryPolicy(max_attempts=3, initial_delay=1.0, backoff=2.0, jitter=0.0)
    assert call_with_retry(flaky, policy=policy, sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]


def test_retry_gives_up_and_reraises() -> None:
    @retry(RetryPolicy(max_attempts=2, jitter=0.0), sleep=lambda _: None)
    def always_fails() -> None:
        raise TimeoutError("nope")

    with pytest.raises(TimeoutError):
        always_fails()


def test_retry_does_not_catch_unlisted_errors() -> None:
    policy = RetryPolicy(max_attempts=5, retry_on=(ConnectionError,))
    with pytest.raises(ValueError):
        call_with_retry(lambda: (_ for _ in ()).throw(ValueError()), policy=policy)


def test_retry_delay_is_capped() -> None:
    policy = RetryPolicy(initial_delay=1.0, backoff=10.0, max_delay=5.0, jitter=0.0)
    assert policy.delay(3) == 5.0


def test_event_bus_subscribe_emit_unsubscribe() -> None:
    bus, seen = EventBus(), []
    unsubscribe = bus.subscribe(Event.STEP_END, lambda **kw: seen.append(kw["step"]))
    bus.emit(Event.STEP_END, step=1)
    unsubscribe()
    bus.emit(Event.STEP_END, step=2)
    assert seen == [1]


def test_run_context_is_a_resettable_singleton() -> None:
    a = RunContext.instance()
    assert RunContext.instance() is a
    RunContext.configure("exp", "runs/exp")
    assert a.run_name == "exp"
    RunContext.reset()
    assert RunContext.instance() is not a
