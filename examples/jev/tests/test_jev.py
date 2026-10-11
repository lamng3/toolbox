"""Tests for the Jev pipeline. AnyJev's ``FakeBackend`` simulates a model with injectable
position bias, so the teacher and evaluation stages are checked without downloading a model."""

from __future__ import annotations

import pytest

anyjev = pytest.importorskip("anyjev")
pytest.importorskip("transformers")

from anyjev.backends.fake import FakeBackend  # noqa: E402
from examples.jev.evaluate import evaluate, expected_calibration_error  # noqa: E402
from examples.jev.teacher import l0_teacher  # noqa: E402

from toolbox.data.repository import load_records  # noqa: E402

FIRST_POSITION_BIAS = [3.0, 0.0]  # strongly prefers whichever option is shown first


def knows_the_answer(records: list[dict]) -> FakeBackend:
    """A fake model that scores the correct option higher, but with a heavy position bias."""
    answers = {r["state"]: r["answer"] for r in records}

    def content(state: str, option: str) -> float:
        return 1.0 if answers.get(state) == option else 0.0

    return FakeBackend(content=content, position_bias=FIRST_POSITION_BIAS)


@pytest.fixture(scope="module")
def records() -> list[dict]:
    seen: dict[str, dict] = {}
    for r in load_records("builtin://decisions.jsonl"):
        if r["options"] == ["yes", "no"]:
            seen.setdefault(r["state"], r)  # one item per state, so state → answer is unambiguous
    return list(seen.values())[:24]


def test_l0_teacher_adds_distributions(records: list[dict]) -> None:
    labelled = l0_teacher(records, knows_the_answer(records))
    for record in labelled:
        assert set(record["probs"]) == set(record["options"])
        assert sum(record["probs"].values()) == pytest.approx(1.0, abs=1e-5)


def test_l0_removes_the_position_bias_that_raw_readout_has(records: list[dict]) -> None:
    backend = knows_the_answer(records)
    raw, l0 = evaluate(backend, records, "raw"), evaluate(backend, records, "L0")
    assert raw["flip_rate"] > 0.5  # the biased model follows the position...
    assert raw["accuracy"] < 0.6  # ...so its single-pass answers are near chance
    assert l0["flip_rate"] == 0.0  # L0 divides the bias out...
    assert l0["accuracy"] == 1.0  # ...and recovers what the model actually knows


def test_expected_calibration_error() -> None:
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert expected_calibration_error([0.9, 0.9], [False, False]) == pytest.approx(0.9)


def test_dataset_uses_anyjev_prompts_and_permutes_options(records: list[dict]) -> None:
    transformers = pytest.importorskip("transformers")
    from anyjev.readout import build_prompt, render_chat, resolve_labels
    from examples.jev.data import AnyJevDecisionDataset, to_question

    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        )
    except OSError:
        pytest.skip("tiny test tokenizer not available offline")
    record = {**records[0], "probs": {"yes": 0.8, "no": 0.2}}
    fixed = AnyJevDecisionDataset([record], tok, permute=False, seed=0)[0]
    q = to_question(record)
    labels, label_ids = resolve_labels(tok, q)
    expected = render_chat(tok, build_prompt(record["state"], q, [0, 1], labels=labels))
    assert fixed["input_ids"] == tok(expected, add_special_tokens=False)["input_ids"]
    assert fixed["label_ids"] == list(label_ids)
    assert fixed["target_probs"] == [0.8, 0.2]
    assert fixed["target"] == q.options.index(record["answer"])
