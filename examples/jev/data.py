"""Decision items → training examples in AnyJev's exact prompt format.

The student is trained on the same prompts AnyJev's ``Decider`` uses at inference (system
message, ``State:`` / ``Question:`` / lettered options, rendered through the model's chat
template, answer read from the next-token logits of the label tokens). The trained model then
works with AnyJev directly, with no adapter in between.

Each example draws a random permutation of the options, which is the training-time counterpart
of AnyJev's permutation-averaged readout: the model cannot learn a position shortcut.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from anyjev import Question
from anyjev.readout import build_prompt, render_chat, resolve_labels
from anyjev.state import render_state

from toolbox.core.constants import DEFAULT_SEED
from toolbox.core.registry import DATASETS
from toolbox.data.datasets.base import DataBundle, ListDataset, train_eval_split
from toolbox.data.repository import Record, load_records
from toolbox.decision.training import DecisionCollator

DEFAULT_EVAL_FRACTION = 0.2


def to_question(record: Mapping[str, Any]) -> Question:
    kind = record.get("kind", "choice")
    if kind == "noul":
        return Question.noul(str(record["question"]))
    return Question.choice(str(record["question"]), list(record["options"]))


def hf_tokenizer(tokenizer: Any) -> Any:
    """Unwrap toolbox's ``HFTokenizer`` adapter to the underlying ``transformers`` tokenizer."""
    return getattr(tokenizer, "hf_tokenizer", tokenizer)


class AnyJevDecisionDataset(ListDataset[Record]):
    """Encodes on access so every epoch sees fresh option permutations."""

    def __init__(self, records: list[Record], tokenizer: Any, permute: bool, seed: int) -> None:
        super().__init__(records)
        self.tok = hf_tokenizer(tokenizer)
        self.permute = permute
        self.rng = random.Random(seed)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.items[index]
        q = to_question(record)
        perm = list(range(q.k))
        if self.permute and record.get("kind", "choice") == "choice":
            self.rng.shuffle(perm)
        labels, label_ids = resolve_labels(self.tok, q)
        prompt = render_chat(
            self.tok, build_prompt(render_state(record.get("state")), q, perm, labels=labels)
        )
        example: dict[str, Any] = {
            "input_ids": self.tok(prompt, add_special_tokens=False)["input_ids"],
            "label_ids": list(label_ids),
        }
        options = list(q.options)
        if "answer" in record:  # position of the correct option under this permutation
            example["target"] = perm.index(options.index(str(record["answer"])))
        if "probs" in record:  # teacher distribution, re-ordered into displayed positions
            example["target_probs"] = [float(record["probs"][options[i]]) for i in perm]
        return example


@DATASETS.register("anyjev_decisions")
def anyjev_decisions(
    tokenizer: Any,
    path: str = "builtin://decisions.jsonl",
    permute: bool = True,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> DataBundle:
    """Records ``{"state", "question", "options", ["answer"], ["probs"]}`` in AnyJev prompts.

    ``probs`` (option → probability) are soft targets from a teacher (see ``teacher.py``);
    ``answer`` is optional when training only from a teacher.
    """
    records = load_records(path)
    train, eval_ = train_eval_split(records, eval_fraction, seed)
    pad_id = hf_tokenizer(tokenizer).pad_token_id
    return DataBundle(
        AnyJevDecisionDataset(train, tokenizer, permute, seed),
        AnyJevDecisionDataset(eval_, tokenizer, False, seed) if eval_ else None,
        DecisionCollator(pad_id),
    )
