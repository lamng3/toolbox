"""Hooks that make existing components emit action records without modifying them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from toolbox.core.registry import CALLBACKS
from toolbox.decision.core import Decision, DecisionReader, Question
from toolbox.tracing.records import ActionKind
from toolbox.tracing.tracer import JsonlTraceStore, Tracer
from toolbox.trainer.callbacks import Callback

if TYPE_CHECKING:
    from toolbox.trainer.base import BaseTrainer


@CALLBACKS.register("trace_rollouts")
class TraceRollouts(Callback):
    """Logs every RL rollout (prompt → completion, reward) as a ``generation`` action.

    The resulting trace log is directly reusable: ``data=traces mode=sft min_reward=1``
    fine-tunes on the policy's own successful samples (expert iteration), and
    ``mode=preference`` builds DPO pairs from same-prompt samples with different rewards.
    """

    def __init__(self, path: str | None = None, agent: str = "policy") -> None:
        self.path, self.agent = path, agent
        self.tracer: Tracer | None = None

    def on_fit_start(self, trainer: BaseTrainer) -> None:
        if trainer.is_main_process:
            root = self.path or f"{trainer.config.output_dir}/traces"
            self.tracer = Tracer(JsonlTraceStore(root))

    def on_rollout_end(self, trainer: BaseTrainer, rollout: Any, rewards: torch.Tensor) -> None:
        if self.tracer is None:
            return
        for prompt, completion, ref, reward in zip(
            rollout.prompts, rollout.completions, rollout.references, rewards.tolist()
        ):
            self.tracer.record(
                ActionKind.GENERATION,
                agent=self.agent,
                state=prompt,
                action=completion,
                reward=reward,
                outcome={"reference": ref},
                metadata={"step": trainer.step},
            )


class TracedDecisionReader:
    """Decorator over :class:`DecisionReader`: every ``decide`` becomes a decision action."""

    def __init__(self, reader: DecisionReader, tracer: Tracer, agent: str = "decider") -> None:
        self.reader, self.tracer, self.agent = reader, tracer, agent

    def decide(self, state: str, question: Question) -> tuple[Decision, str]:
        """Returns the decision and the action id (pass it to ``tracer.feedback`` later)."""
        with self.tracer.span(
            ActionKind.DECISION,
            agent=self.agent,
            state={"state": state, "question": question.text},
            options=list(question.options),
        ) as record:
            decision = self.reader.decide(state, question)
            record.action, record.probs = decision.answer, decision.probs
            record.metadata["question_kind"] = question.kind.value
        return decision, record.id
