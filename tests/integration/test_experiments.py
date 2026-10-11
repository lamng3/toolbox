"""Every packaged CPU experiment must run end-to-end (2 steps) — the "does it plug together"
contract that keeps the toolbox usable as a template."""

from __future__ import annotations

from pathlib import Path

import pytest

from toolbox.experiment import load_config, run_experiment

CPU_EXPERIMENTS = [
    "sft_tiny",
    "dpo_tiny",
    "reward_model_tiny",
    "grpo_tiny",
    "decision_tiny",
    "decision_rl_tiny",
    "ppo_cartpole",
    "gnn_pretrain",
    "lm_pretrain_tiny",
]
GPU_EXPERIMENTS = ["grpo_qwen_gsm8k", "sft_qwen", "dpo_qwen"]


def _overrides(name: str, out: Path) -> list[str]:
    return [
        f"experiment={name}",
        "trainer.max_steps=2",
        "trainer.device=cpu",
        f"output_dir={out}",
        "trainer.eval_batches=1",
    ]


@pytest.mark.slow
@pytest.mark.parametrize("name", CPU_EXPERIMENTS)
def test_experiment_runs(name: str, tmp_path: Path) -> None:
    metrics = run_experiment(load_config(_overrides(name, tmp_path / name)))
    assert metrics, f"{name} returned no metrics"
    assert (tmp_path / name / "checkpoints" / "last.pt").exists()


@pytest.mark.slow
def test_pretrain_then_finetune_gnn(tmp_path: Path) -> None:
    run_experiment(load_config(_overrides("gnn_pretrain", tmp_path / "pre")))
    ckpt = tmp_path / "pre" / "checkpoints" / "last.pt"
    metrics = run_experiment(
        load_config(
            [
                *_overrides("gnn_finetune", tmp_path / "ft"),
                f"model.pretrained_checkpoint={ckpt}",
            ]
        )
    )
    assert "eval/accuracy" in metrics


@pytest.mark.slow
def test_sft_then_grpo_then_sft_on_traces(tmp_path: Path) -> None:
    run_experiment(load_config(_overrides("sft_tiny", tmp_path / "sft")))
    ckpt = tmp_path / "sft" / "checkpoints" / "last.pt"
    run_experiment(
        load_config([*_overrides("grpo_tiny", tmp_path / "grpo"), f"model.init_from={ckpt}"])
    )
    traces = tmp_path / "grpo" / "traces"
    assert (traces / "actions.jsonl").exists()
    metrics = run_experiment(
        load_config(
            [
                *_overrides("sft_from_traces", tmp_path / "it"),
                f"data.path={traces}",
                "data.min_reward=null",
            ]
        )
    )
    assert "loss" in metrics


@pytest.mark.parametrize("name", GPU_EXPERIMENTS)
def test_gpu_experiment_configs_compose(name: str) -> None:
    cfg = load_config([f"experiment={name}"])
    assert cfg.model.name == "hf_causal_lm" and cfg.tokenizer.name == "hf"
