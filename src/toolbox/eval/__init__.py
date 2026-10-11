"""Standalone evaluators — run against a model or engine outside of training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from toolbox.core.registry import EVALUATORS
from toolbox.data.repository import load_records
from toolbox.decision.core import DecisionReader, Question
from toolbox.rewards import Reward
from toolbox.serving.base import InferenceEngine, SamplingParams


@EVALUATORS.register("generation_reward")
def generation_reward(
    engine: InferenceEngine,
    reward: Reward,
    prompts: Sequence[str],
    references: Sequence[str],
    params: SamplingParams | None = None,
) -> dict[str, float]:
    """Mean reward of greedy completions (e.g. accuracy on a math benchmark)."""
    params = params or SamplingParams(max_new_tokens=64, temperature=0.0)
    completions = engine.generate(prompts, params)
    scores = reward(prompts, [c.text for c in completions], references)
    return {"reward": float(scores.mean()), "num_examples": float(len(prompts))}


@EVALUATORS.register("decision_accuracy")
def decision_accuracy(
    reader: DecisionReader, path: str = "builtin://decisions.jsonl", limit: int | None = None
) -> dict[str, float]:
    """Accuracy plus *flip rate*: how often the answer changes when options are reversed —
    a direct measure of position bias (lower is better)."""
    records: list[dict[str, Any]] = load_records(path)[:limit]
    correct = flips = 0
    for r in records:
        question = Question.choice(r["question"], r["options"])
        reversed_q = Question.choice(r["question"], list(reversed(r["options"])))
        answer = reader.decide(r.get("state", ""), question).answer
        flipped = reader.decide(r.get("state", ""), reversed_q).answer
        correct += answer == str(r["answer"])
        flips += answer != flipped
    n = max(len(records), 1)
    return {"accuracy": correct / n, "flip_rate": flips / n, "num_examples": float(len(records))}


__all__ = ["decision_accuracy", "generation_reward"]
