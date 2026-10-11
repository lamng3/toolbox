"""Turn action traces into training data — the loop from *acting* to *learning*.

``traces`` builds one of three datasets from a trace log:

* ``mode=sft`` — behaviour cloning / rejection sampling: ``state → action`` pairs, keeping
  only actions whose reward ≥ ``min_reward`` (expert iteration, STaR).
* ``mode=decision`` — decision records become ``{state, question, options, answer, probs}``
  examples for the decision objectives (``decision_ce`` / ``decision_distill``).
* ``mode=preference`` — actions taken from the *same state* with different rewards become
  ``{prompt, chosen, rejected}`` pairs for DPO or reward modelling.
"""

from __future__ import annotations

import itertools
import json
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from toolbox.core.constants import DEFAULT_SEED
from toolbox.core.registry import DATASETS
from toolbox.data.datasets.base import DataBundle
from toolbox.data.repository import JsonlRepository, Record
from toolbox.data.tokenization import Tokenizer
from toolbox.tracing.records import ActionKind, ActionRecord
from toolbox.tracing.tracer import DEFAULT_TRACE_KEY, TraceStore

MODES = ("sft", "decision", "preference")


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def to_sft_records(
    records: Iterable[ActionRecord], min_reward: float | None = None
) -> list[Record]:
    out = []
    for r in records:
        if not r.succeeded or r.action is None:
            continue
        if min_reward is not None and (r.reward is None or r.reward < min_reward):
            continue
        out.append({"prompt": _text(r.state), "response": " " + _text(r.action)})
    return out


def to_decision_records(
    records: Iterable[ActionRecord], min_reward: float | None = None
) -> list[Record]:
    out = []
    for r in records:
        if r.kind != ActionKind.DECISION.value or not r.options or r.action not in r.options:
            continue
        if min_reward is not None and (r.reward is None or r.reward < min_reward):
            continue
        state = r.state if isinstance(r.state, dict) else {"state": _text(r.state)}
        example: Record = {
            "state": state.get("state", ""),
            "question": state.get("question", r.metadata.get("question", "Which option?")),
            "options": list(r.options),
            "answer": r.action,
        }
        if r.probs and set(r.probs) == set(r.options):
            example["probs"] = r.probs
        out.append(example)
    return out


def to_preference_records(records: Iterable[ActionRecord], min_margin: float = 0.0) -> list[Record]:
    by_state: dict[str, list[ActionRecord]] = defaultdict(list)
    for r in records:
        if r.reward is not None and r.action is not None and r.succeeded:
            by_state[_text(r.state)].append(r)
    pairs = []
    for state, group in by_state.items():
        for a, b in itertools.combinations(group, 2):
            if a.reward is None or b.reward is None or abs(a.reward - b.reward) <= min_margin:
                continue
            chosen, rejected = (a, b) if a.reward > b.reward else (b, a)
            pairs.append(
                {
                    "prompt": state,
                    "chosen": " " + _text(chosen.action),
                    "rejected": " " + _text(rejected.action),
                }
            )
    return pairs


CONVERTERS = {
    "sft": lambda recs, min_reward: to_sft_records(recs, min_reward),
    "decision": lambda recs, min_reward: to_decision_records(recs, min_reward),
    "preference": lambda recs, min_reward: to_preference_records(recs),
}


@DATASETS.register("traces")
def traces(
    tokenizer: Tokenizer,
    path: str,
    mode: str = "sft",
    key: str = DEFAULT_TRACE_KEY,
    agent: str | None = None,
    kind: str | None = None,
    min_reward: float | None = None,
    eval_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> DataBundle:
    """Build a training set from ``<path>/<key>.jsonl`` traces (see module docstring)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    records = list(TraceStore(JsonlRepository(path), key).latest_by_id().values())
    records = [
        r
        for r in records
        if (agent is None or r.agent == agent) and (kind is None or r.kind == kind)
    ]
    examples = CONVERTERS[mode](records, min_reward)
    if not examples:
        raise ValueError(f"No usable '{mode}' examples in traces at {path}")

    # Reuse the regular dataset factories so tokenisation stays in one place.
    from toolbox.data.datasets.text import preference, sft
    from toolbox.decision.training import decisions

    factories: dict[str, Callable[..., DataBundle]] = {
        "sft": sft,
        "preference": preference,
        "decision": decisions,
    }
    factory = factories[mode]
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "examples.jsonl"
        JsonlRepository(tmp).write("examples", examples)
        return factory(tokenizer=tokenizer, path=str(file), eval_fraction=eval_fraction, seed=seed)


def summarize(store: TraceStore) -> dict[str, Any]:
    """Counts, error rate, mean reward and latency per (agent, kind) — for ``toolbox traces``."""
    groups: dict[tuple[str, str], list[ActionRecord]] = defaultdict(list)
    for r in store.latest_by_id().values():
        groups[(r.agent, str(r.kind))].append(r)
    summary = {}
    for (agent, kind), recs in sorted(groups.items()):
        rewards = [r.reward for r in recs if r.reward is not None]
        latencies = [r.latency_ms for r in recs if r.latency_ms is not None]
        summary[f"{agent}/{kind}"] = {
            "count": len(recs),
            "errors": sum(not r.succeeded for r in recs),
            "rewarded": len(rewards),
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        }
    return summary
