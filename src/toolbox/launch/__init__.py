"""Launching experiments locally, with torchrun, or on SLURM (pattern: *Command*).

Each way of running an experiment is a :class:`LaunchCommand` object with ``describe``
(what would happen — the dry run), ``execute`` and, where meaningful, ``undo`` (cancel a
submitted job). The CLI only builds a command from config and hands it to a
:class:`CommandInvoker`, which keeps a history — so "submit 5 seeds, then cancel them all" is
just ``invoker.undo_all()``.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

from toolbox.core.logging import get_logger
from toolbox.core.registry import LAUNCHERS
from toolbox.core.retry import RetryPolicy, call_with_retry

TEMPLATE_DIR = Path(__file__).parent / "templates"
SBATCH_TEMPLATE = TEMPLATE_DIR / "slurm.sbatch"
JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")
SUBMIT_RETRY = RetryPolicy(
    max_attempts=3, initial_delay=5.0, retry_on=(subprocess.CalledProcessError,)
)
DEFAULT_MASTER_PORT = 29500
PREEMPT_SIGNAL_SECONDS = 120

log = get_logger("launch")
Runner = Callable[[Sequence[str]], str]


def run_subprocess(cmd: Sequence[str]) -> str:
    return subprocess.run(list(cmd), check=True, capture_output=True, text=True).stdout


def _config_dir_args(config_dir: str | None) -> list[str]:
    return ["--config-dir", str(Path(config_dir).resolve())] if config_dir else []


@dataclass
class LaunchResult:
    launcher: str
    detail: str
    job_id: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class LaunchCommand(ABC):
    name: str = "launch"

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of what ``execute`` will do (dry run)."""

    @abstractmethod
    def execute(self) -> LaunchResult: ...

    def undo(self) -> None:
        """Revert ``execute`` when possible (e.g. cancel a submitted job)."""
        raise NotImplementedError(f"{type(self).__name__} cannot be undone")


@LAUNCHERS.register("local")
class LocalTrainCommand(LaunchCommand):
    """Train in the current process."""

    name = "local"

    def __init__(self, overrides: Sequence[str] = (), config_dir: str | None = None) -> None:
        self.overrides, self.config_dir = list(overrides), config_dir

    def describe(self) -> str:
        return f"run in-process: toolbox train {shlex.join(self.overrides)}"

    def execute(self) -> LaunchResult:
        from toolbox.experiment import load_config, run_experiment

        metrics = run_experiment(load_config(self.overrides, self.config_dir))
        return LaunchResult(self.name, "completed", metrics=metrics)


@LAUNCHERS.register("torchrun")
class TorchrunCommand(LaunchCommand):
    """Single-node multi-GPU training: ``torchrun --nproc-per-node N``."""

    name = "torchrun"

    def __init__(
        self,
        overrides: Sequence[str] = (),
        nproc_per_node: int = 1,
        config_dir: str | None = None,
        runner: Runner = run_subprocess,
    ) -> None:
        self.overrides, self.nproc, self.runner = list(overrides), nproc_per_node, runner
        self.config_dir = config_dir

    def command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={self.nproc}",
            "-m",
            "toolbox.cli",
            "train",
            *_config_dir_args(self.config_dir),
            *self.overrides,
            "strategy=ddp",
        ]

    def describe(self) -> str:
        return shlex.join(self.command())

    def execute(self) -> LaunchResult:
        return LaunchResult(self.name, self.runner(self.command()))


@dataclass
class SlurmSpec:
    """Resources and environment of a SLURM job; mirrors ``configs/launcher/slurm.yaml``."""

    job_name: str = "toolbox"
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    nodes: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 8
    mem: str = "64G"
    time: str = "04:00:00"
    array: str | None = None
    requeue: bool = True
    setup: Sequence[str] = ("source .venv/bin/activate",)
    env: dict[str, str] = field(default_factory=dict)
    extra_directives: Sequence[str] = ()
    master_port: int = DEFAULT_MASTER_PORT


@LAUNCHERS.register("slurm")
class SlurmSubmitCommand(LaunchCommand):
    """Render an ``sbatch`` script from the template and submit it.

    ``runner`` is injectable so tests (and ``--dry-run``) never touch a real cluster.
    """

    name = "slurm"

    def __init__(
        self,
        overrides: Sequence[str] = (),
        output_dir: str = "runs/slurm",
        config_dir: str | None = None,
        runner: Runner = run_subprocess,
        **spec: Any,
    ) -> None:
        self.overrides = list(overrides)
        self.config_dir = config_dir
        self.spec = SlurmSpec(**spec)
        self.output_dir = Path(output_dir)
        self.runner = runner
        self.job_id: str | None = None

    def render(self) -> str:
        s = self.spec
        optional = {
            "partition": s.partition,
            "account": s.account,
            "qos": s.qos,
            "constraint": s.constraint,
            "array": s.array,
        }
        directives = [f"#SBATCH --{k}={v}" for k, v in optional.items() if v]
        if s.requeue:
            directives.append("#SBATCH --requeue")
        directives += [f"#SBATCH {d}" for d in s.extra_directives]
        overrides = [
            *_config_dir_args(self.config_dir),
            *self.overrides,
            "strategy=ddp",
            "trainer.resume_from=auto",
        ]
        return Template(SBATCH_TEMPLATE.read_text()).substitute(
            job_name=s.job_name,
            nodes=s.nodes,
            gpus_per_node=s.gpus_per_node,
            cpus_per_task=s.cpus_per_task,
            mem=s.mem,
            time=s.time,
            log_dir=self.output_dir / "logs",
            preempt_signal_seconds=PREEMPT_SIGNAL_SECONDS,
            optional_directives="\n".join(directives),
            setup="\n".join(s.setup),
            env_exports="\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in s.env.items()),
            master_port=s.master_port,
            omp_threads=max(1, s.cpus_per_task // s.gpus_per_node),
            overrides=shlex.join(overrides),
        )

    def script_path(self) -> Path:
        return self.output_dir / f"{self.spec.job_name}.sbatch"

    def describe(self) -> str:
        return self.render()

    def execute(self) -> LaunchResult:
        path = self.script_path()
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        path.write_text(self.render())
        os.chmod(path, 0o755)
        stdout = call_with_retry(self.runner, ["sbatch", str(path)], policy=SUBMIT_RETRY)
        match = JOB_ID_PATTERN.search(stdout)
        self.job_id = match.group(1) if match else None
        log.info("Submitted %s as job %s", path, self.job_id)
        return LaunchResult(self.name, stdout.strip(), job_id=self.job_id)

    def undo(self) -> None:
        if self.job_id is None:
            raise RuntimeError("nothing to cancel: job was not submitted")
        self.runner(["scancel", self.job_id])
        log.info("Cancelled job %s", self.job_id)


class CommandInvoker:
    """Executes commands and remembers them so they can be undone in reverse order."""

    def __init__(self) -> None:
        self.history: list[LaunchCommand] = []

    def run(self, command: LaunchCommand, dry_run: bool = False) -> LaunchResult | str:
        if dry_run:
            return command.describe()
        result = command.execute()
        self.history.append(command)
        return result

    def undo_last(self) -> None:
        if self.history:
            self.history.pop().undo()

    def undo_all(self) -> None:
        while self.history:
            self.undo_last()


__all__ = [
    "CommandInvoker",
    "LaunchCommand",
    "LaunchResult",
    "LocalTrainCommand",
    "SlurmSpec",
    "SlurmSubmitCommand",
    "TorchrunCommand",
]
