# Decision models

A *decision model* answers a typed question (choice, yes/no or score) with a **probability per
option**. These are the numbers you need to automate a decision with a known error rate or to
escalate when the model is unsure. The approach follows
[AnyJev](https://github.com/nokia-applied-research/AnyJev): the decision is read from the
next-token distribution over option labels instead of being parsed from generated text.

```python
from toolbox.decision import DecisionReader, Question

reader = DecisionReader(model, tokenizer, rotations=None)   # None = all K rotations
decision = reader.decide(
    state="Ticket: card was charged twice for one purchase.",
    question=Question.choice("Which queue?", ["billing", "fraud", "technical"]),
)
decision.answer   # "billing"
decision.probs    # {"billing": 0.81, "fraud": 0.15, "technical": 0.04}
```

## Removing position bias

LLMs prefer some option positions and labels ("A") regardless of content. The reader scores
every **cyclic rotation** of the options, maps each rotation's label probabilities back to the
original options, and averages the log-probabilities. A pure position bias then cancels out.
The unit test `test_rotation_averaging_removes_position_bias` checks exactly this: a model that
always says "A" comes out uniform.

`toolbox.eval.decision_accuracy` reports accuracy and **flip rate**, the fraction of questions
whose answer changes when the options are reversed. Lower flip rate means less position bias.

## Training

| `objective=` | Signal | Use when |
| --- | --- | --- |
| `decision_ce` | correct option | labelled decisions |
| `decision_distill` | soft target distribution `probs` | self-distillation, e.g. from the model's CoT answers or a debiased multi-rotation read ("Tacit"-style) |
| `decision_pg` | reward of a sampled option | only outcomes are observed (RL) |

```bash
toolbox train experiment=decision_tiny
toolbox train experiment=decision_tiny objective=decision_distill   # needs `probs` in data
toolbox train experiment=decision_rl_tiny
```

Training data randomly rotates the options of each example (`augment_rotations=true`), so the
model cannot learn a position shortcut.

## Training a Jev model end to end

[`examples/jev`](https://github.com/lamng3/toolbox/tree/main/examples/jev) is a full pipeline that
uses [AnyJev](https://github.com/nokia-applied-research/AnyJev) for the readout and evaluation,
and toolbox for training. A teacher labels decisions with no human labels needed; a student LLM
learns to give that decision in one forward pass, in AnyJev's exact prompt format, so AnyJev can
serve it directly.

## Data

```json
{"state": "Numbers a=11, b=16.", "question": "Is 11 greater than 16?",
 "options": ["yes", "no"], "answer": "no", "probs": {"yes": 0.1, "no": 0.9}}
```

Logged agent decisions convert to this format directly. See [Agent tracing](tracing.md).
