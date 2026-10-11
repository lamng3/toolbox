from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from toolbox.launch import CommandInvoker, SlurmSubmitCommand, TorchrunCommand


class FakeCluster:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Sequence[str]) -> str:
        self.calls.append(list(cmd))
        return "Submitted batch job 4242\n" if cmd[0] == "sbatch" else ""


def test_slurm_script_rendering(tmp_path: Path) -> None:
    command = SlurmSubmitCommand(
        ["experiment=sft_tiny"],
        output_dir=str(tmp_path),
        partition="gpu",
        account="lab",
        nodes=2,
        gpus_per_node=4,
    )
    script = command.describe()
    for expected in (
        "#SBATCH --partition=gpu",
        "#SBATCH --account=lab",
        "#SBATCH --nodes=2",
        "--nproc-per-node=4",
        "experiment=sft_tiny",
        "trainer.resume_from=auto",
        "scontrol requeue $SLURM_JOB_ID",
        "#SBATCH --requeue",
    ):
        assert expected in script
    assert "${" not in script.replace("${toolbox", "")  # all placeholders substituted


def test_slurm_submit_and_cancel(tmp_path: Path) -> None:
    cluster = FakeCluster()
    command = SlurmSubmitCommand(["experiment=sft_tiny"], output_dir=str(tmp_path), runner=cluster)
    invoker = CommandInvoker()
    result = invoker.run(command)
    assert command.script_path().exists()
    assert getattr(result, "job_id", None) == "4242"
    invoker.undo_all()
    assert cluster.calls[-1] == ["scancel", "4242"]


def test_dry_run_does_not_execute(tmp_path: Path) -> None:
    cluster = FakeCluster()
    out = CommandInvoker().run(
        SlurmSubmitCommand(output_dir=str(tmp_path), runner=cluster), dry_run=True
    )
    assert isinstance(out, str) and cluster.calls == []


def test_torchrun_command() -> None:
    cmd = TorchrunCommand(["experiment=sft_tiny"], nproc_per_node=8).command()
    assert "--nproc-per-node=8" in cmd and cmd[-1] == "strategy=ddp"
