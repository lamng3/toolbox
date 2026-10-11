from __future__ import annotations

from pathlib import Path

import torch

from toolbox.experiment import ExperimentBuilder
from toolbox.trainer import Callback, CheckpointManager, EarlyStopping


def _builder(run_dir: Path, steps: int = 4, **trainer: object) -> ExperimentBuilder:
    return (
        ExperimentBuilder("unit", output_dir=str(run_dir))
        .with_tokenizer("byte")
        .with_data("sft", max_length=48)
        .with_model("tiny_gpt", d_model=32, n_layers=1, n_heads=4, max_seq_len=64)
        .with_objective("sft")
        .with_trainer(
            "supervised", max_steps=steps, batch_size=4, device="cpu", eval_every=0, **trainer
        )
        .with_callbacks("jsonl")
    )


def test_fit_writes_checkpoint_config_and_metrics(run_dir: Path) -> None:
    metrics = _builder(run_dir).build().fit()
    assert "loss" in metrics and "eval/loss" in metrics
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "config.yaml").exists()
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) >= 4


def test_resume_continues_from_checkpoint(run_dir: Path) -> None:
    _builder(run_dir, steps=3).build().fit()
    trainer = _builder(run_dir, steps=5, resume_from="auto").build()
    trainer.fit()
    assert trainer.step == 5


def test_loss_decreases_on_tiny_sft(run_dir: Path) -> None:
    trainer = _builder(run_dir, steps=40).with_optimizer("adamw", lr=3e-3).build()
    first = trainer.train_step()["loss"]
    trainer.fit()
    assert trainer.last_metrics["loss"] < first


def test_callbacks_can_stop_training(run_dir: Path) -> None:
    class StopAtTwo(Callback):
        def on_step_end(self, trainer, step, metrics) -> None:  # type: ignore[no-untyped-def]
            if step == 2:
                trainer.request_stop("enough")

    builder = _builder(run_dir, steps=10)
    trainer = builder.build()
    StopAtTwo().attach(trainer.events)
    trainer.fit()
    assert trainer.step == 2


def test_early_stopping_counts_bad_evals() -> None:
    class FakeTrainer:
        stopped: str | None = None

        def request_stop(self, reason: str) -> None:
            self.stopped = reason

    es, trainer = EarlyStopping(patience=2), FakeTrainer()
    for value in (1.0, 0.9, 0.95, 0.97):
        es.on_eval_end(trainer, 0, {"eval/loss": value})  # type: ignore[arg-type]
    assert trainer.stopped is not None


def test_checkpoint_rotation(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, keep_last=2)
    for step in range(1, 5):
        manager.save({"step": step, "w": torch.zeros(1)}, step)
    assert [p.name for p in manager.step_checkpoints()] == ["step_3.pt", "step_4.pt"]
    assert manager.load(manager.latest())["step"] == 4  # type: ignore[arg-type]
