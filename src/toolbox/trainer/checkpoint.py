"""Checkpoint storage (pattern: *Repository*).

The trainer asks for "save state at step N" / "latest state"; the manager decides file names,
rotation and atomic writes. A cloud implementation only needs the same three methods.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

from toolbox.core.constants import CHECKPOINT_DIRNAME, LAST_CHECKPOINT_NAME

STEP_PATTERN = re.compile(r"step_(\d+)\.pt$")


class CheckpointManager:
    """Saves ``step_<N>.pt`` plus ``last.pt`` under ``<output_dir>/checkpoints``."""

    def __init__(self, output_dir: str | Path, keep_last: int = 2) -> None:
        self.dir = Path(output_dir) / CHECKPOINT_DIRNAME
        self.keep_last = keep_last

    def save(self, state: dict[str, Any], step: int) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"step_{step}.pt"
        self._atomic_save(state, path)
        self._atomic_save(state, self.dir / LAST_CHECKPOINT_NAME)
        self._rotate()
        return path

    def latest(self) -> Path | None:
        last = self.dir / LAST_CHECKPOINT_NAME
        return last if last.exists() else None

    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        state: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
        return state

    def step_checkpoints(self) -> list[Path]:
        found = [
            (int(m.group(1)), p)
            for p in self.dir.glob("step_*.pt")
            if (m := STEP_PATTERN.search(p.name))
        ]
        return [p for _, p in sorted(found)]

    def _rotate(self) -> None:
        if self.keep_last <= 0:
            return
        for stale in self.step_checkpoints()[: -self.keep_last]:
            stale.unlink(missing_ok=True)

    @staticmethod
    def _atomic_save(state: dict[str, Any], path: Path) -> None:
        """Write to a temp file then rename, so a preempted SLURM job never leaves a torn file."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
