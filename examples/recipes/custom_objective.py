"""Add a technique without touching the toolbox: register a new objective and train with it.

The same pattern works for models, datasets, rewards, callbacks, engines ... For a reusable
package, expose the module under the ``toolbox.plugins`` entry-point group instead (see the
"Extending" docs page) and it becomes available to the CLI and configs automatically.

    python -m examples.recipes.custom_objective
"""

from __future__ import annotations

import torch
from torch import nn

from toolbox.core.registry import OBJECTIVES
from toolbox.core.types import Batch, LossOutput
from toolbox.experiment import ExperimentBuilder
from toolbox.objectives.base import SupervisedObjective
from toolbox.objectives.utils import shift_for_next_token, token_logprobs


@OBJECTIVES.register("sft_with_entropy_bonus")
class SFTWithEntropyBonus(SupervisedObjective):
    """SFT loss minus a small entropy bonus (keeps the policy from collapsing early)."""

    def __init__(self, entropy_coef: float = 0.01) -> None:
        self.entropy_coef = entropy_coef

    def __call__(self, model: nn.Module, batch: Batch) -> LossOutput:
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        logits, labels = shift_for_next_token(logits, batch["labels"])
        mask = (labels >= 0).float()
        nll = -(token_logprobs(logits, labels) * mask).sum() / mask.sum()
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(-1)
        entropy = (entropy * mask).sum() / mask.sum()
        return LossOutput(nll - self.entropy_coef * entropy, {"entropy": entropy.item()})


def main() -> None:
    trainer = (
        ExperimentBuilder("custom_objective")
        .with_tokenizer("byte")
        .with_data("sft", max_length=64)
        .with_model("tiny_gpt", d_model=64, n_layers=2, n_heads=4)
        .with_objective("sft_with_entropy_bonus", entropy_coef=0.02)
        .with_trainer("supervised", max_steps=30, batch_size=16, device="cpu", eval_every=0)
        .build()
    )
    print(trainer.fit())


if __name__ == "__main__":
    main()
