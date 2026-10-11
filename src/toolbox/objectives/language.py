"""Language-modelling objectives."""

from __future__ import annotations

import math

import torch
from torch import nn

from toolbox.core.registry import OBJECTIVES
from toolbox.core.types import Batch, LossOutput
from toolbox.objectives.base import SupervisedObjective
from toolbox.objectives.utils import cross_entropy_lm


@OBJECTIVES.register("lm_ce")
@OBJECTIVES.register("sft")
class CausalLMLoss(SupervisedObjective):
    """Next-token cross-entropy. With prompt labels masked (``IGNORE_INDEX``) this is SFT."""

    def __call__(self, model: nn.Module, batch: Batch) -> LossOutput:
        logits = model(batch["input_ids"], attention_mask=batch.get("attention_mask"))
        loss = cross_entropy_lm(logits, batch["labels"])
        with torch.no_grad():
            ppl = math.exp(min(float(loss), 20.0))  # cap avoids overflow early in training
        return LossOutput(loss, {"perplexity": ppl})
