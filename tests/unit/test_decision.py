from __future__ import annotations

import pytest
import torch
from torch import nn

from toolbox.data.tokenization import ByteTokenizer
from toolbox.decision import DecisionReader, Question, decisions
from toolbox.decision.core import rotate
from toolbox.decision.training import DecisionCrossEntropy, DecisionDistillation
from toolbox.nn.models import TinyGPT


class LabelBiasedModel(nn.Module):
    """Always prefers label 'A' regardless of content — pure position bias."""

    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = torch.zeros(*input_ids.shape, self.vocab)
        logits[..., ord("A")] = 5.0
        return logits + self.dummy


def test_question_constructors() -> None:
    assert Question.yes_no("ok?").options == ("yes", "no")
    assert Question.score("rate", levels=3, first_level=0).options == ("0", "1", "2")
    with pytest.raises(ValueError):
        Question.choice("one option", ["only"])


def test_rotation() -> None:
    assert rotate(["a", "b", "c"], 1) == ["b", "c", "a"]


def test_rotation_averaging_removes_position_bias() -> None:
    tok = ByteTokenizer()
    question = Question.choice("Pick", ["x", "y", "z"])
    biased = DecisionReader(LabelBiasedModel(tok.vocab_size), tok, rotations=1)
    debiased = DecisionReader(LabelBiasedModel(tok.vocab_size), tok, rotations=None)
    assert biased.decide("", question).answer == "x"  # always the first slot
    probs = debiased.distribution("", question)
    torch.testing.assert_close(probs, torch.full((3,), 1 / 3))


def test_decision_objectives_run_on_dataset() -> None:
    tok = ByteTokenizer()
    bundle = decisions(tok, eval_fraction=0.0)
    batch = bundle.collate_fn([bundle.train[i] for i in range(4)])
    model = TinyGPT(tok.vocab_size, d_model=32, n_layers=1, n_heads=4, max_seq_len=128)
    for objective in (DecisionCrossEntropy(), DecisionDistillation()):
        out = objective(model, batch)
        assert torch.isfinite(out.loss) and 0 <= out.metrics["accuracy"] <= 1
