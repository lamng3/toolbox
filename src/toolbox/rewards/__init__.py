"""Reward functions for online RL (pattern: *Strategy*, composed with *Composite*).

A reward scores completions: ``reward(prompts, completions, references) -> Tensor[B]``.
Verifiable rewards (exact match, format) need no model; learned rewards wrap a reward model;
an LLM-as-judge reward talks to any :class:`~toolbox.serving.InferenceEngine`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from toolbox.core.registry import ENGINES, REWARDS
from toolbox.data.collators import pad_sequences

NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
JUDGE_SCORE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")


class Reward(ABC):
    """Scores a batch of completions. Higher is better."""

    @abstractmethod
    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], references: Sequence[str]
    ) -> torch.Tensor: ...


def extract_last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text)
    return matches[-1] if matches else None


@REWARDS.register("exact_match")
class ExactMatchReward(Reward):
    """1.0 if the extracted answer equals the reference, else 0.0.

    Args:
        extract: ``"last_number"`` (math word problems) or ``"strip"`` (whole string).
    """

    EXTRACTORS = {
        "last_number": extract_last_number,
        "strip": lambda s: s.strip(),
        "first_token": lambda s: (s.split() or [""])[0],
    }

    def __init__(self, extract: str = "last_number") -> None:
        self.extract = self.EXTRACTORS[extract]

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], references: Sequence[str]
    ) -> torch.Tensor:
        return torch.tensor(
            [
                float(self.extract(c) is not None and self.extract(c) == self.extract(r))
                for c, r in zip(completions, references)
            ]
        )


@REWARDS.register("format")
class FormatReward(Reward):
    """1.0 if the completion matches ``pattern`` (e.g. ``<think>.*</think><answer>.*</answer>``)."""

    def __init__(self, pattern: str, full_match: bool = False) -> None:
        self.regex = re.compile(pattern, re.DOTALL)
        self.full_match = full_match

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], references: Sequence[str]
    ) -> torch.Tensor:
        match = self.regex.fullmatch if self.full_match else self.regex.search
        return torch.tensor([float(match(c) is not None) for c in completions])


@REWARDS.register("length_penalty")
class LengthPenalty(Reward):
    """``-|len(completion) - target| / target`` in characters — discourages rambling."""

    def __init__(self, target_chars: int = 200) -> None:
        self.target = target_chars

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], references: Sequence[str]
    ) -> torch.Tensor:
        return torch.tensor([-abs(len(c) - self.target) / self.target for c in completions])


@REWARDS.register("weighted_sum")
class WeightedSumReward(Reward):
    """Composite reward: ``Σ wᵢ · rewardᵢ``. Children are built from config specs.

    Example config::

        reward:
          name: weighted_sum
          components:
            - {name: exact_match, weight: 1.0}
            - {name: format, pattern: "A: ", weight: 0.1}
    """

    WEIGHT_KEY = "weight"

    def __init__(self, components: Sequence[Mapping[str, Any]]) -> None:
        self.children: list[tuple[float, Reward]] = []
        for spec in components:
            spec = dict(spec)
            weight = float(spec.pop(self.WEIGHT_KEY, 1.0))
            self.children.append((weight, REWARDS.build(spec)))

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], references: Sequence[str]
    ) -> torch.Tensor:
        total = torch.zeros(len(completions))
        for weight, reward in self.children:
            total += weight * reward(prompts, completions, references)
        return total


@REWARDS.register("reward_model")
class ModelReward(Reward):
    """Learned reward from a toolbox reward-model checkpoint (trained with ``reward_bt``)."""

    def __init__(
        self, checkpoint: str, model: Mapping[str, Any], tokenizer: Any, device: str = "cpu"
    ) -> None:
        from toolbox.core.registry import MODELS

        self.tokenizer, self.device = tokenizer, torch.device(device)
        self.model = MODELS.build(model, inject={"vocab_size": tokenizer.vocab_size})
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state.get("model", state))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], references: Sequence[str]
    ) -> torch.Tensor:
        ids = [
            self.tokenizer.encode(p + c, add_bos=True, add_eos=True)
            for p, c in zip(prompts, completions)
        ]
        input_ids = pad_sequences(ids, self.tokenizer.pad_token_id).to(self.device)
        mask = (input_ids != self.tokenizer.pad_token_id).long()
        return self.model(input_ids, attention_mask=mask).float().cpu()


@REWARDS.register("llm_judge")
class LLMJudgeReward(Reward):
    """LLM-as-a-judge: asks an inference engine to grade each completion on ``[0, max_score]``.

    The judge runs behind any engine adapter (SGLang, vLLM, OpenAI-compatible HTTP, HF).
    """

    DEFAULT_TEMPLATE = (
        "Grade the answer to the question on a scale from 0 to {max_score}. "
        "Reply with the number only.\n\nQuestion: {prompt}\nReference: {reference}\n"
        "Answer: {completion}\nScore:"
    )

    def __init__(
        self, engine: Mapping[str, Any], max_score: float = 10.0, template: str | None = None
    ) -> None:
        from toolbox.serving import SamplingParams

        self.engine = ENGINES.build(engine)
        self.max_score = max_score
        self.template = template or self.DEFAULT_TEMPLATE
        self.params = SamplingParams(max_new_tokens=8, temperature=0.0)

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], references: Sequence[str]
    ) -> torch.Tensor:
        queries = [
            self.template.format(prompt=p, completion=c, reference=r, max_score=self.max_score)
            for p, c, r in zip(prompts, completions, references)
        ]
        outputs = self.engine.generate(queries, self.params)
        scores = []
        for out in outputs:
            match = JUDGE_SCORE_PATTERN.search(out.text)
            score = float(match.group(1)) if match else 0.0
            scores.append(min(score, self.max_score) / self.max_score)
        return torch.tensor(scores)


__all__ = [
    "ExactMatchReward",
    "FormatReward",
    "LLMJudgeReward",
    "LengthPenalty",
    "ModelReward",
    "Reward",
    "WeightedSumReward",
    "extract_last_number",
]
