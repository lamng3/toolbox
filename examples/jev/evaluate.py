"""Evaluate decision models through AnyJev, so numbers are comparable with AnyJev's own.

Readouts compared:

* ``raw`` — one forward pass, options in the given order (what a Tacit-style student serves);
* ``L0`` — AnyJev's permutation-debiased readout (``K`` forward passes).

Metrics: accuracy (if gold ``answer`` exists), agreement with the teacher (if ``probs`` exist),
**flip rate** (answer changes when the options are reversed: position bias), expected
calibration error, and **forward passes per decision** (the cost a one-pass student saves;
wall-clock time is a poor proxy because AnyJev shares prompt prefixes across permutations).
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from anyjev import Decider, Question

from examples.jev.data import to_question
from toolbox.data.repository import Record

ECE_BINS = 10
MS_PER_S = 1000.0


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = ECE_BINS
) -> float:
    conf, hit = np.asarray(confidences), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(hit[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def reversed_question(q: Question) -> Question:
    return Question.choice(q.text, list(reversed(q.options)))


def evaluate(backend: Any, records: Sequence[Record], level: str) -> dict[str, float]:
    """Run AnyJev's ``Decider`` at ``level`` over ``records`` and summarise."""
    decider = Decider(backend, level=level)
    answers, flips, confidences, correct, agree, passes = [], [], [], [], [], []
    start = time.perf_counter()
    for record in records:
        q = to_question(record)
        decision = decider.decide(record.get("state"), [q])[0]
        answers.append(decision.argmax)
        passes.append(int(decision.diagnostics.get("permutations", 1)))
        confidences.append(decision.confidence)
        if q.kind == "choice":
            flipped = decider.decide(record.get("state"), [reversed_question(q)])[0]
            flips.append(flipped.argmax != decision.argmax)
        if "answer" in record:
            correct.append(decision.argmax == str(record["answer"]))
        if "probs" in record:
            agree.append(decision.argmax == max(record["probs"], key=record["probs"].get))
    elapsed = time.perf_counter() - start
    calls = len(records) + len(flips)
    metrics = {
        "decisions": float(len(records)),
        "forward_passes": float(np.mean(passes)),
        "latency_ms": elapsed / max(calls, 1) * MS_PER_S,
        "flip_rate": float(np.mean(flips)) if flips else float("nan"),
    }
    if correct:
        metrics["accuracy"] = float(np.mean(correct))
        metrics["ece"] = expected_calibration_error(confidences, correct)
    if agree:
        metrics["teacher_agreement"] = float(np.mean(agree))
    return metrics
