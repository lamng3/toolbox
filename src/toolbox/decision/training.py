"""Training decision models: dataset, collator and objectives.

Three ways to train the same readout:

* ``decision_ce`` — supervised: cross-entropy against the correct option.
* ``decision_distill`` — self-distillation (Tacit-style): match soft target distributions,
  e.g. produced by the same model when it reasons (CoT) or by a debiased multi-rotation read.
* ``decision_pg`` — RL: sample an option, get reward 1 if correct (or any verifier), and apply
  REINFORCE with a batch baseline. Useful when only outcomes, not labels, are observed.
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from toolbox.core.constants import DEFAULT_SEED
from toolbox.core.registry import DATASETS, OBJECTIVES
from toolbox.core.types import Batch, LossOutput
from toolbox.data.collators import PaddingCollator, pad_sequences
from toolbox.data.datasets.base import DataBundle, ListDataset, train_eval_split
from toolbox.data.repository import load_records
from toolbox.data.tokenization import Tokenizer
from toolbox.decision.core import label_logprobs, label_token_ids, render_prompt, rotate
from toolbox.objectives.base import SupervisedObjective

PAD_OPTION = -1


class DecisionCollator:
    """Pads prompts (right) and per-example option label ids / soft targets."""

    def __init__(self, pad_token_id: int) -> None:
        self.base = PaddingCollator(pad_token_id)

    def __call__(self, examples: list[dict[str, Any]]) -> Batch:
        label_ids = [ex.pop("label_ids") for ex in examples]
        soft = [ex.pop("target_probs", None) for ex in examples]
        batch = self.base(examples)
        batch["label_ids"] = pad_sequences(label_ids, PAD_OPTION)
        if all(s is not None for s in soft):
            width = batch["label_ids"].shape[1]
            batch["target_probs"] = torch.tensor(
                [list(s) + [0.0] * (width - len(s)) for s in soft if s is not None]
            )
        return batch


class _DecisionDataset(ListDataset[dict[str, Any]]):
    """Encodes on access so each epoch can see a fresh random rotation of the options."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Tokenizer,
        augment_rotations: bool,
        seed: int,
    ) -> None:
        super().__init__(records)
        self.tokenizer = tokenizer
        self.augment = augment_rotations
        self.rng = random.Random(seed)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.items[index]
        options = list(record["options"])
        shift = self.rng.randrange(len(options)) if self.augment else 0
        rotated = rotate(options, shift)
        prompt = render_prompt(record.get("state", ""), record["question"], rotated)
        example: dict[str, Any] = {
            "input_ids": self.tokenizer.encode(prompt, add_bos=True),
            "label_ids": label_token_ids(self.tokenizer, len(options)),
            "target": rotated.index(str(record["answer"])),
        }
        if "probs" in record:  # soft targets keyed by option text
            example["target_probs"] = [float(record["probs"][o]) for o in rotated]
        return example


@DATASETS.register("decisions")
def decisions(
    tokenizer: Tokenizer,
    path: str = "builtin://decisions.jsonl",
    augment_rotations: bool = True,
    eval_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> DataBundle:
    """Records ``{"state", "question", "options", "answer", ["probs"]}``.

    ``augment_rotations`` randomly rotates options during training so the model cannot learn a
    position shortcut — the training-time counterpart of the rotation-debiased readout.
    """
    train, eval_ = train_eval_split(load_records(path), eval_fraction, seed)
    return DataBundle(
        _DecisionDataset(train, tokenizer, augment_rotations, seed),
        _DecisionDataset(eval_, tokenizer, False, seed) if eval_ else None,
        DecisionCollator(tokenizer.pad_token_id),
    )


def _option_logprobs(model: nn.Module, batch: Batch) -> torch.Tensor:
    return label_logprobs(model, batch["input_ids"], batch["attention_mask"], batch["label_ids"])


def _accuracy(logps: torch.Tensor, target: torch.Tensor) -> float:
    return float((logps.argmax(-1) == target).float().mean())


@OBJECTIVES.register("decision_ce")
class DecisionCrossEntropy(SupervisedObjective):
    def __call__(self, model: nn.Module, batch: Batch) -> LossOutput:
        logps = _option_logprobs(model, batch)
        loss = F.nll_loss(logps, batch["target"])
        return LossOutput(loss, {"accuracy": _accuracy(logps, batch["target"])})


@OBJECTIVES.register("decision_distill")
class DecisionDistillation(SupervisedObjective):
    """KL(target ‖ model) over options; falls back to one-hot targets when none are given."""

    def __call__(self, model: nn.Module, batch: Batch) -> LossOutput:
        logps = _option_logprobs(model, batch)
        target = batch.get("target_probs")
        if target is None:
            target = F.one_hot(batch["target"], logps.shape[-1]).float()
        safe_logps = logps.masked_fill(torch.isinf(logps), 0.0)
        kl = (target * (torch.log(target.clamp(min=1e-12)) - safe_logps)).sum(-1).mean()
        return LossOutput(kl, {"accuracy": _accuracy(logps, batch["target"])})


@OBJECTIVES.register("decision_pg")
class DecisionPolicyGradient(SupervisedObjective):
    """REINFORCE over options with a mean-reward baseline and entropy bonus."""

    def __init__(self, entropy_coef: float = 0.01) -> None:
        self.entropy_coef = entropy_coef

    def __call__(self, model: nn.Module, batch: Batch) -> LossOutput:
        logps = _option_logprobs(model, batch)
        dist = torch.distributions.Categorical(logits=logps)
        action = dist.sample()
        reward = (action == batch["target"]).float()
        advantage = reward - reward.mean()
        entropy = dist.entropy().mean()
        loss = -(advantage * dist.log_prob(action)).mean() - self.entropy_coef * entropy
        return LossOutput(
            loss,
            {
                "reward": float(reward.mean()),
                "accuracy": _accuracy(logps, batch["target"]),
            },
        )
