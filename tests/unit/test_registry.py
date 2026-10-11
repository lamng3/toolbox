from __future__ import annotations

import importlib.metadata
import sys
import types
from pathlib import Path

import pytest

from toolbox.core.registry import MODELS, OBJECTIVES, Registry, RegistryError, load_plugins


def test_register_and_build_with_overrides() -> None:
    registry: Registry[object] = Registry("thing")

    @registry.register("pair")
    def make_pair(a: int, b: int = 2) -> tuple[int, int]:
        return (a, b)

    assert registry.build({"name": "pair", "a": 1}) == (1, 2)
    assert registry.build({"name": "pair", "a": 1}, b=5) == (1, 5)
    assert registry.build("pair", a=3) == (3, 2)


def test_inject_only_passes_accepted_and_unset_kwargs() -> None:
    registry: Registry[object] = Registry("thing")

    @registry.register("needs_a")
    def needs_a(a: int) -> int:
        return a

    @registry.register("takes_all")
    def takes_all(**kwargs: int) -> dict[str, int]:
        return kwargs

    inject = {"a": 1, "unrelated": 9}
    assert registry.build("needs_a", inject=inject) == 1
    assert registry.build({"name": "needs_a", "a": 7}, inject=inject) == 7  # spec wins
    assert registry.build("takes_all", inject=inject) == inject


def test_unknown_name_lists_available() -> None:
    with pytest.raises(RegistryError, match=r"Available:.*tiny_gpt"):
        MODELS.get("does_not_exist")


def test_duplicate_registration_is_rejected() -> None:
    registry: Registry[object] = Registry("thing")
    registry.add("x", object)
    with pytest.raises(RegistryError, match="already registered"):
        registry.add("x", object)
    registry.add("x", dict, override=True)
    assert registry.get("x") is dict


def test_builtins_are_registered() -> None:
    for name in ("sft", "dpo", "grpo", "ppo_clip", "decision_ce", "masked_feature"):
        assert name in OBJECTIVES


def test_plugins_load_from_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("fake_toolbox_plugin")

    def _register() -> None:
        OBJECTIVES.add("plugin_loss", object, override=True)

    module.register = _register  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_toolbox_plugin", module)
    entry_point = importlib.metadata.EntryPoint(
        name="fake", value="fake_toolbox_plugin:register", group="toolbox.plugins"
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [entry_point])
    assert load_plugins() == ["fake"]


def test_config_imports_register_project_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from toolbox.experiment import ExperimentBuilder

    (tmp_path / "my_project_losses.py").write_text(
        "from toolbox.core.registry import OBJECTIVES\n"
        "OBJECTIVES.register('my_project_loss')(lambda: 'built')\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    builder = ExperimentBuilder.from_config(
        {"imports": ["my_project_losses"], "objective": {"name": "my_project_loss"}}
    )
    assert builder.objective() == "built"
    assert "imports" not in builder.extra
