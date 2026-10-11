"""Trace an agent's decisions as atomic actions, then train on them.

1. A decision model answers typed questions; every answer is logged as a ``decision`` action
   with its option probabilities (``TracedDecisionReader``).
2. When the outcome is known, ``tracer.feedback(action_id, reward=...)`` attaches it.
3. The trace log becomes a training set (``data=traces mode=decision``) — here keeping only
   rewarded (correct) decisions, i.e. learning from the agent's own successes.

    python -m examples.recipes.agent_tracing --out runs/agent_tracing
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from toolbox.data.repository import load_records
from toolbox.data.tokenization import ByteTokenizer
from toolbox.decision import DecisionReader, Question
from toolbox.experiment import ExperimentBuilder
from toolbox.nn.models import TinyGPT
from toolbox.tracing import JsonlTraceStore, TracedDecisionReader, Tracer, summarize

MODEL: dict[str, Any] = {"d_model": 64, "n_layers": 2, "n_heads": 4, "max_seq_len": 256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/agent_tracing"))
    parser.add_argument("--episodes", type=int, default=64)
    args = parser.parse_args()

    tokenizer = ByteTokenizer()
    store = JsonlTraceStore(str(args.out / "traces"))
    tracer = Tracer(store)
    agent = TracedDecisionReader(
        DecisionReader(TinyGPT(tokenizer.vocab_size, **MODEL), tokenizer), tracer, agent="router"
    )

    # 1-2. Act, then attach delayed feedback.
    for record in load_records("builtin://decisions.jsonl")[: args.episodes]:
        decision, action_id = agent.decide(
            record["state"], Question.choice(record["question"], record["options"])
        )
        tracer.feedback(action_id, reward=float(decision.answer == record["answer"]))
    for key, row in summarize(store).items():
        print(key, row)

    # 3. Learn from the rewarded decisions.
    trainer = (
        ExperimentBuilder("agent_tracing", output_dir=str(args.out / "train"))
        .with_tokenizer("byte")
        .with_data("traces", path=str(args.out / "traces"), mode="decision", min_reward=1.0)
        .with_model("tiny_gpt", **MODEL)
        .with_objective("decision_ce")
        .with_trainer("supervised", max_steps=50, batch_size=8, device="cpu", eval_every=0)
        .build()
    )
    print(trainer.fit())


if __name__ == "__main__":
    main()
