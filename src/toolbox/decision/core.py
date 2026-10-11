"""Decision models: an LLM answers a typed question with a probability per option.

Inspired by AnyJev (https://github.com/nokia-applied-research/AnyJev): instead of parsing
generated text, the decision is *read* from the next-token distribution over option labels
(``A``, ``B``, ...). Because LLMs prefer some positions and labels regardless of content, the
:class:`DecisionReader` scores every cyclic rotation of the options and averages the
log-probabilities per option, which cancels position/label bias without any labels.
"""

from __future__ import annotations

import enum
import string
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from toolbox.data.collators import pad_sequences
from toolbox.data.tokenization import Tokenizer

LABELS = string.ascii_uppercase
YES_NO_OPTIONS = ("yes", "no")


class QuestionKind(str, enum.Enum):
    CHOICE = "choice"
    YES_NO = "yes_no"
    SCORE = "score"


@dataclass(frozen=True)
class Question:
    """A typed question. ``score`` questions use ordered levels as options."""

    text: str
    options: tuple[str, ...]
    kind: QuestionKind = QuestionKind.CHOICE

    @classmethod
    def choice(cls, text: str, options: Sequence[str]) -> Question:
        return cls(text, tuple(options), QuestionKind.CHOICE)

    @classmethod
    def yes_no(cls, text: str) -> Question:
        return cls(text, YES_NO_OPTIONS, QuestionKind.YES_NO)

    @classmethod
    def score(cls, text: str, levels: int, first_level: int = 1) -> Question:
        options = tuple(str(first_level + i) for i in range(levels))
        return cls(text, options, QuestionKind.SCORE)

    def __post_init__(self) -> None:
        if not 2 <= len(self.options) <= len(LABELS):
            raise ValueError(f"need 2..{len(LABELS)} options, got {len(self.options)}")


@dataclass
class Decision:
    answer: str
    probs: dict[str, float]
    rotations: int = 1
    metadata: dict[str, object] = field(default_factory=dict)


def render_prompt(state: str, question: str, options: Sequence[str]) -> str:
    """The canonical decision prompt; the model's next token should be an option label."""
    lines = [state.strip(), f"Question: {question.strip()}", "Options:"]
    lines += [f"{LABELS[i]}. {opt}" for i, opt in enumerate(options)]
    lines.append("Answer: ")
    return "\n".join(lines)


def label_token_ids(tokenizer: Tokenizer, num_options: int) -> list[int]:
    """Token id of each option label; labels must be single tokens for the readout."""
    ids = []
    for label in LABELS[:num_options]:
        encoded = tokenizer.encode(label)
        if len(encoded) != 1:
            raise ValueError(f"label {label!r} is not a single token for this tokenizer")
        ids.append(encoded[0])
    return ids


def rotate(options: Sequence[str], shift: int) -> list[str]:
    shift %= len(options)
    return list(options[shift:]) + list(options[:shift])


def label_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_ids: torch.Tensor,
) -> torch.Tensor:
    """Log-softmax over the option labels at the last real token of each prompt.

    Args:
        input_ids: right-padded prompts ``(B, T)``.
        attention_mask: ``(B, T)``, 1 for real tokens.
        label_ids: ``(B, K)`` label token ids, ``-1`` for padding options.

    Returns:
        ``(B, K)`` log-probabilities renormalised over the valid options.
    """
    logits = model(input_ids, attention_mask=attention_mask)
    last = attention_mask.long().sum(dim=1) - 1
    final = logits[torch.arange(logits.shape[0], device=logits.device), last].float()
    valid = label_ids >= 0
    option_logits = final.gather(1, label_ids.clamp(min=0))
    option_logits = option_logits.masked_fill(~valid, -torch.inf)
    return torch.log_softmax(option_logits, dim=-1)


class DecisionReader:
    """Reads calibrated option distributions from any causal LM.

    Args:
        rotations: number of cyclic rotations to average (``None`` = all ``K``; ``1`` = the
            raw, biased single readout).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        rotations: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.rotations = rotations
        self.device = torch.device(device)

    @torch.no_grad()
    def distribution(self, state: str, question: Question) -> torch.Tensor:
        """``(K,)`` probabilities aligned with ``question.options``."""
        options = list(question.options)
        k = len(options)
        shifts = range(k if self.rotations is None else min(self.rotations, k))
        prompts = [render_prompt(state, question.text, rotate(options, s)) for s in shifts]
        ids = [self.tokenizer.encode(p, add_bos=True) for p in prompts]
        input_ids = pad_sequences(ids, self.tokenizer.pad_token_id).to(self.device)
        mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels = torch.tensor(
            [label_token_ids(self.tokenizer, k)] * len(prompts), device=self.device
        )
        logps = label_logprobs(self.model, input_ids, mask, labels)  # (R, K) in rotated order
        # Rotated position j holds original option (j + s) % K: undo the rotation.
        aligned = torch.stack([logps[r].roll(shifts=s) for r, s in enumerate(shifts)])
        return torch.softmax(aligned.mean(dim=0), dim=-1).cpu()

    def decide(self, state: str, question: Question) -> Decision:
        probs = self.distribution(state, question)
        answer = question.options[int(probs.argmax())]
        return Decision(
            answer=answer,
            probs={o: float(p) for o, p in zip(question.options, probs)},
            rotations=self.rotations or len(question.options),
        )
