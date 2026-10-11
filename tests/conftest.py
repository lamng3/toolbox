"""Shared fixtures: every test gets a clean global state and a temporary run directory."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from toolbox.core.logging import RunContext
from toolbox.tracing import set_tracer


@pytest.fixture(autouse=True)
def _isolate_globals() -> Iterator[None]:
    RunContext.reset()
    set_tracer(None)
    torch.manual_seed(0)
    yield
    RunContext.reset()
    set_tracer(None)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run"
