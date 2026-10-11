"""The TRL backend, end to end with a tiny Hugging Face model.

TRL ≥ 1.0 computes log-probs with a Triton kernel for SFT/DPO/GRPO, so those algorithms need a
Linux GPU machine. They skip elsewhere; reward-model training runs everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

trl = pytest.importorskip("trl")
pytest.importorskip("datasets")

from toolbox.experiment import ExperimentBuilder  # noqa: E402

TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


CASES = {
    "reward": ("preference", "builtin://preference.jsonl", None, {"max_length": 64}),
    "sft": ("sft", "builtin://sft.jsonl", None, {"max_length": 64}),
    "dpo": ("preference", "builtin://preference.jsonl", None, {"max_length": 64}),
    "grpo": (
        "prompts",
        "builtin://arithmetic.jsonl",
        "exact_match",
        {"num_generations": 4, "max_completion_length": 8},
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", sorted(CASES))
def test_trl_algorithms_run_from_toolbox_config(algorithm: str, tmp_path: Path) -> None:
    if algorithm != "reward" and not triton_available():
        pytest.skip("TRL's SFT/DPO/GRPO kernels need Triton (Linux + GPU)")
    data, path, reward, args = CASES[algorithm]
    builder = (
        ExperimentBuilder(f"trl_{algorithm}", output_dir=str(tmp_path))
        .with_data(data, path=path)
        .with_model("hf_causal_lm", model_name_or_path=TINY_MODEL)
        .with_optimizer("adamw", lr=1e-4)
        .with_trainer(
            "trl",
            algorithm=algorithm,
            max_steps=2,
            batch_size=4,
            device="cpu",
            log_every=1,
            args=args,
        )
        .with_callbacks("jsonl")
    )
    if reward:
        builder.with_reward(reward)
    try:
        metrics = builder.build().fit()
    except OSError:
        pytest.skip("tiny test model not available offline")
    assert "loss" in metrics
    assert (tmp_path / "final").exists() and (tmp_path / "metrics.jsonl").exists()


def test_unknown_trl_options_are_rejected(tmp_path: Path) -> None:
    builder = (
        ExperimentBuilder("trl_bad", output_dir=str(tmp_path))
        .with_data("sft", path="builtin://sft.jsonl")
        .with_model("hf_causal_lm", model_name_or_path=TINY_MODEL)
        .with_trainer("trl", algorithm="sft", max_steps=1, args={"not_a_trl_option": 1})
    )
    with pytest.raises(ValueError, match="no option"):
        builder.build()._trl_args(trl)
