"""The atomic unit of agent behaviour: an :class:`ActionRecord`.

Every observable thing an agent does — a decision between options, a tool call, a text
generation, an environment step — becomes one immutable-by-convention record with its input
(the *state* the agent saw), its output (the *action*), how it was produced (option
probabilities, model, latency, tokens) and, later, how it turned out (*reward* / *outcome*).

Because the schema is uniform, a log of records doubles as a training set: see
:mod:`toolbox.tracing.datasets` for converting traces into SFT, decision and preference data.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any


class ActionKind(str, enum.Enum):
    DECISION = "decision"  # choose among typed options (see toolbox.decision)
    GENERATION = "generation"  # free-form text produced by a model
    TOOL_CALL = "tool_call"  # call into an external tool / API
    ENV_STEP = "env_step"  # action in an RL environment
    STEP = "step"  # a generic unit of work (e.g. a whole agent turn)


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class ActionRecord:
    """One atomic action.

    Attributes:
        kind: what sort of action (see :class:`ActionKind`).
        agent: logical name of the actor (``"planner"``, ``"router"`` ...).
        state: what the agent observed — prompt, context, environment observation.
        action: what it did — chosen option, generated text, tool arguments.
        options: candidate actions for decisions.
        probs: probability per option (decisions) — the model's calibrated belief.
        reward: scalar feedback attached immediately or later via ``Tracer.feedback``.
        outcome: free-form result (tool output, verifier verdict, error message).
        trace_id / parent_id: group records into a tree (episode → turn → tool call).
        latency_ms, tokens, model: cost/profiling information.
        metadata: anything else (prompt version, temperature, git sha ...).
    """

    kind: ActionKind | str
    agent: str = "agent"
    state: Any = None
    action: Any = None
    options: list[str] | None = None
    probs: dict[str, float] | None = None
    reward: float | None = None
    outcome: Any = None
    trace_id: str = field(default_factory=new_id)
    parent_id: str | None = None
    id: str = field(default_factory=new_id)
    timestamp: float = field(default_factory=time.time)
    latency_ms: float | None = None
    tokens: int | None = None
    model: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value if isinstance(self.kind, ActionKind) else self.kind
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def succeeded(self) -> bool:
        return self.error is None
