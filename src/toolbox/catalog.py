"""A Markdown catalog of every registered component and config preset.

The docs site renders this at build time (``docs/gen_ref_pages.py``), so the catalog can never
drift from the code. Adding a component with ``@REGISTRY.register(...)`` or a YAML under
``configs/`` is enough for it to show up.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from toolbox.core.registry import ALL_REGISTRIES, load_builtins
from toolbox.experiment import CONFIG_DIR

SOURCE_URL = "https://github.com/lamng3/toolbox/blob/main/src/"
NO_SUMMARY = "—"


def summary(factory: Any) -> str:
    """First paragraph of a factory's docstring, on one line."""
    doc = inspect.getdoc(factory) or ""
    first = doc.split("\n\n")[0].replace("\n", " ").strip()
    return first.replace("|", "\\|") or NO_SUMMARY


def source_link(factory: Any) -> str:
    module = inspect.getmodule(factory)
    if module is None or not module.__name__.startswith("toolbox"):
        return ""
    path = module.__name__.replace(".", "/")
    path += "/__init__.py" if Path(str(module.__file__)).name == "__init__.py" else ".py"
    return f"[`{module.__name__}`]({SOURCE_URL}{path})"


def registry_tables() -> list[str]:
    load_builtins()
    lines: list[str] = []
    for kind, registry in sorted(ALL_REGISTRIES.items()):
        lines += [f"### `{kind}`", "", "| Name | Summary | Source |", "| --- | --- | --- |"]
        for name in registry.names():
            factory = registry.get(name)
            lines.append(f"| `{name}` | {summary(factory)} | {source_link(factory)} |")
        lines.append("")
    return lines


def preset_tables(config_dir: Path = CONFIG_DIR) -> list[str]:
    lines: list[str] = []
    for group in sorted(p for p in config_dir.iterdir() if p.is_dir()):
        presets = sorted(group.glob("*.yaml"))
        names = ", ".join(f"`{p.stem}`" for p in presets)
        lines.append(f"| `{group.name}=` | {names} |")
    return ["| Group | Presets |", "| --- | --- |", *lines, ""]


def render_markdown() -> str:
    return "\n".join(
        [
            "# Component catalog",
            "",
            "Every component you can select by name, and every config preset. This page is "
            "generated from the registries and `src/toolbox/configs/` when the docs are built, "
            "so it always matches the code. `toolbox list` prints the same names in the terminal.",
            "",
            "## Config presets",
            "",
            "Select with `group=preset`, e.g. `toolbox train experiment=grpo_tiny reward=learned`.",
            "",
            *preset_tables(),
            "## Registered components",
            "",
            "Use the name as `name:` in a config section, or with the matching "
            "`ExperimentBuilder.with_*` method.",
            "",
            *registry_tables(),
        ]
    )
