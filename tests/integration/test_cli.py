from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from toolbox.cli import app

runner = CliRunner()


def test_list() -> None:
    result = runner.invoke(app, ["list", "objective"])
    assert result.exit_code == 0 and "grpo" in result.output


def test_config_shows_composed_experiment() -> None:
    result = runner.invoke(app, ["config", "experiment=dpo_tiny"])
    assert result.exit_code == 0 and "name: dpo" in result.output


def test_launch_slurm_dry_run(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "launch",
            "experiment=sft_tiny",
            "launcher=slurm",
            "launcher.partition=gpu",
            f"output_dir={tmp_path}",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "#SBATCH --partition=gpu" in result.output


def test_env() -> None:
    result = runner.invoke(app, ["env"])
    assert result.exit_code == 0 and "torch" in result.output
