"""Teachers: soft decision targets without human labels.

* ``l0`` — AnyJev's training-free readout (``Decider(level="L0")``): option probabilities
  averaged over permutations with label/position bias divided out. Distilling it teaches the
  student to give, in **one** forward pass, what L0 gets from ``K``.
* ``cot`` — the model reasons step by step before answering. The distribution over its final
  answers (``samples`` reasoning chains) is the target. This mirrors how AnyJev describes its Tacit
  models: one forward pass reproducing the answer the model reaches when it reasons.

Both return the input records with an added ``probs`` field (option → probability).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import torch
from anyjev import Decider
from anyjev.readout import LETTERS, build_prompt, resolve_labels
from anyjev.state import render_state

from examples.jev.data import to_question
from toolbox.data.repository import Record

COT_SYSTEM = (
    "You are a careful decision maker. Think step by step, then finish with a final line of "
    "the form 'Answer: <letter>'."
)
COT_ANSWER = re.compile(r"Answer:\s*\(?([A-Z])\)?", re.IGNORECASE)
SMOOTHING = 0.5  # add-½ smoothing keeps every option's probability > 0


def l0_teacher(records: Sequence[Record], backend: Any, **decider_kwargs: Any) -> list[Record]:
    """Label records with AnyJev's L0 (permutation-debiased) distributions."""
    decider = Decider(backend, level="L0", **decider_kwargs)
    labelled = []
    for record in records:
        q = to_question(record)
        decision = decider.decide(record.get("state"), [q])[0]
        labelled.append({**record, "probs": decision.distribution})
    return labelled


@torch.no_grad()
def cot_teacher(
    records: Sequence[Record],
    model: Any,
    tokenizer: Any,
    samples: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> list[Record]:
    """Label records with the empirical distribution of chain-of-thought final answers."""
    labelled = []
    device = next(model.parameters()).device
    for record in records:
        q = to_question(record)
        labels, _ = resolve_labels(tokenizer, q)
        spec = build_prompt(render_state(record.get("state")), q, list(range(q.k)), labels=labels)
        user = spec.user.replace("Answer with the letter only.", "Explain briefly, then answer.")
        messages = [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt] * samples, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        texts = tokenizer.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        votes: Counter[int] = Counter()
        for text in texts:
            match = COT_ANSWER.findall(text)
            if match and match[-1].upper() in LETTERS[: q.k]:
                votes[LETTERS.index(match[-1].upper())] += 1
        total = sum(votes.values()) + SMOOTHING * q.k
        probs = {opt: (votes[i] + SMOOTHING) / total for i, opt in enumerate(q.options)}
        labelled.append({**record, "probs": probs, "cot_parsed": sum(votes.values())})
    return labelled
