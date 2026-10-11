"""Regenerate the tiny offline datasets shipped in ``src/toolbox/data/builtin``.

They exist so every example runs on a laptop with no downloads. Run:
    python scripts/make_builtin_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "toolbox" / "data" / "builtin"
SEED = 0
N_ARITHMETIC = 512
N_SFT = 256
N_PREFERENCE = 256
N_DECISIONS = 256
MAX_OPERAND = 20

CORPUS = """\
A language model assigns probabilities to sequences of tokens. Pretraining teaches it to predict the next token on a large corpus of text.

Post-training adapts a pretrained model to follow instructions, prefer helpful answers, and reason. Supervised fine-tuning, preference optimisation and reinforcement learning are the common tools.

Supervised fine-tuning imitates demonstrations. The loss is the cross-entropy of the response tokens given the prompt; prompt tokens are masked out.

Direct preference optimisation learns from pairs of answers. It increases the likelihood of the chosen answer relative to the rejected one, measured against a frozen reference model.

Group relative policy optimisation samples several answers per prompt, scores them with a reward function, and normalises rewards within the group to obtain advantages. No value network is needed.

A graph neural network passes messages along edges. Each layer aggregates the features of neighbours and updates the representation of every node.

Self-supervised graph pretraining masks node features and asks the network to reconstruct them, or contrasts two augmented views of the same graph.

A decision model reads the probability of each option from the next-token distribution. Reading every rotation of the options removes position bias.

Inference engines such as SGLang and vLLM batch requests continuously and cache key-value states to serve models with high throughput.

Experiments should be reproducible: fix the seed, save the resolved configuration, and record the code version with every checkpoint.
"""

TOPICS = [
    "sorting",
    "hashing",
    "recursion",
    "graphs",
    "caching",
    "testing",
    "profiling",
    "batching",
]
DECISION_KINDS = [
    ("Is {a} greater than {b}?", ["yes", "no"], lambda a, b: "yes" if a > b else "no"),
    (
        "Is {a} + {b} even or odd?",
        ["even", "odd"],
        lambda a, b: "even" if (a + b) % 2 == 0 else "odd",
    ),
    ("Which is larger: {a} or {b}?", None, lambda a, b: str(max(a, b))),
]


def write_jsonl(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tiny_corpus.txt").write_text(CORPUS, encoding="utf-8")

    arithmetic = []
    for _ in range(N_ARITHMETIC):
        a, b = rng.randint(0, MAX_OPERAND), rng.randint(0, MAX_OPERAND)
        arithmetic.append({"prompt": f"Q: {a}+{b}=? A:", "answer": str(a + b)})
    write_jsonl("arithmetic.jsonl", arithmetic)

    write_jsonl(
        "sft.jsonl",
        [{"prompt": r["prompt"], "response": f" {r['answer']}"} for r in arithmetic[:N_SFT]],
    )

    preference = []
    for r in arithmetic[N_SFT : N_SFT + N_PREFERENCE]:
        wrong = str(int(r["answer"]) + rng.choice([-2, -1, 1, 2]))
        preference.append(
            {"prompt": r["prompt"], "chosen": f" {r['answer']}", "rejected": f" {wrong}"}
        )
    write_jsonl("preference.jsonl", preference)

    decisions = []
    for _ in range(N_DECISIONS):
        template, options, rule = rng.choice(DECISION_KINDS)
        a, b = rng.randint(0, MAX_OPERAND), rng.randint(0, MAX_OPERAND)
        if a == b:
            b += 1
        opts = options or [str(a), str(b)]
        decisions.append(
            {
                "state": f"Numbers a={a}, b={b}.",
                "question": template.format(a=a, b=b),
                "options": opts,
                "answer": rule(a, b),
            }
        )
    write_jsonl("decisions.jsonl", decisions)


if __name__ == "__main__":
    main()
