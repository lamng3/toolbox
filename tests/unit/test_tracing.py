from __future__ import annotations

from pathlib import Path

import pytest

from toolbox.data.tokenization import ByteTokenizer
from toolbox.tracing import (
    ActionKind,
    InMemoryTraceStore,
    JsonlTraceStore,
    Tracer,
    current_action,
    summarize,
    to_decision_records,
    to_preference_records,
    to_sft_records,
    traces,
)


def test_spans_nest_and_record_latency() -> None:
    store = InMemoryTraceStore()
    tracer = Tracer(store)
    with tracer.span(ActionKind.STEP, agent="planner") as outer:
        with tracer.span(ActionKind.TOOL_CALL, agent="search") as inner:
            assert current_action() is inner
        inner_id = inner.id
    records = {r.id: r for r in store.records()}
    assert records[inner_id].parent_id == outer.id
    assert records[inner_id].trace_id == outer.trace_id
    assert records[outer.id].latency_ms is not None
    assert current_action() is None


def test_span_records_errors() -> None:
    store = InMemoryTraceStore()
    tracer = Tracer(store)
    with pytest.raises(RuntimeError), tracer.span("tool_call"):
        raise RuntimeError("tool down")
    (record,) = store.records()
    assert record.error == "RuntimeError: tool down" and not record.succeeded


def test_traced_decorator_and_feedback() -> None:
    store = InMemoryTraceStore()
    tracer = Tracer(store)

    @tracer.traced(agent="adder")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    (record,) = store.records()
    assert record.state == {"args": [2, 3], "kwargs": {}} and record.action == 5
    tracer.feedback(record.id, reward=1.0)
    assert store.latest_by_id()[record.id].reward == 1.0


def test_disabled_tracer_records_nothing() -> None:
    store = InMemoryTraceStore()
    Tracer(store, enabled=False).record("step")
    assert store.records() == []


def _populate(tracer: Tracer) -> None:
    for action, reward in (("4", 1.0), ("5", 0.0), ("4", 1.0)):
        tracer.record(ActionKind.GENERATION, state="Q: 2+2=? A:", action=action, reward=reward)
    tracer.record(
        ActionKind.DECISION,
        state={"state": "s", "question": "q?"},
        options=["yes", "no"],
        action="yes",
        probs={"yes": 0.8, "no": 0.2},
    )


def test_conversions() -> None:
    store = InMemoryTraceStore()
    _populate(Tracer(store))
    records = store.records()
    assert len(to_sft_records(records, min_reward=1.0)) == 2
    pairs = to_preference_records(records)
    assert pairs and all(p["chosen"] == " 4" and p["rejected"] == " 5" for p in pairs)
    (decision,) = to_decision_records(records)
    assert decision["answer"] == "yes" and decision["probs"]["yes"] == 0.8


def test_traces_dataset_and_summary(tmp_path: Path) -> None:
    store = JsonlTraceStore(str(tmp_path))
    _populate(Tracer(store))
    bundle = traces(ByteTokenizer(), str(tmp_path), mode="sft", min_reward=1.0, eval_fraction=0.0)
    assert len(bundle.train) == 2
    summary = summarize(store)
    assert summary["agent/generation"]["count"] == 3
    assert summary["agent/generation"]["mean_reward"] == pytest.approx(2 / 3)
