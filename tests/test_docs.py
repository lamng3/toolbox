"""Docs must follow the code. These tests fail CI when something is added without docs.

The component catalog (``toolbox.catalog``) is generated at build time, so registry names are
always listed. These checks cover what generation can't: docstrings, hand-written guides,
navigation and the examples index.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from toolbox.catalog import NO_SUMMARY, render_markdown
from toolbox.cli import app
from toolbox.experiment import CONFIG_DIR

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples"
PACKAGE = ROOT / "src" / "toolbox"
GENERATED_PAGES = {"concepts/components.md"}  # written by docs/gen_ref_pages.py


def docs_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in DOCS.rglob("*.md"))


def test_every_component_has_a_docstring_summary() -> None:
    undocumented = [line for line in render_markdown().splitlines() if f"| {NO_SUMMARY} |" in line]
    assert not undocumented, f"add docstrings for: {undocumented}"


@pytest.mark.parametrize(
    "experiment", sorted(p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml"))
)
def test_every_experiment_is_documented(experiment: str) -> None:
    assert experiment in docs_text(), f"experiment '{experiment}' is not mentioned in docs/"


def test_every_cli_command_is_documented() -> None:
    text = docs_text()
    commands = [
        c.name or (c.callback.__name__ if c.callback else "") for c in app.registered_commands
    ]
    missing = [c for c in commands if f"toolbox {c}" not in text]
    assert not missing, f"CLI commands without docs: {missing}"


def test_every_package_is_in_the_architecture_page() -> None:
    layout = (DOCS / "concepts" / "architecture.md").read_text(encoding="utf-8")
    packages = [p.name for p in PACKAGE.iterdir() if p.is_dir() and (p / "__init__.py").exists()]
    modules = [p.name for p in PACKAGE.glob("*.py") if p.name != "__init__.py"]
    missing = [name for name in packages + modules if name not in layout]
    assert not missing, f"add to docs/concepts/architecture.md: {missing}"


def test_every_docs_page_is_in_the_nav() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    pages = {p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.md")}
    listed = set(re.findall(r":\s*([\w/.-]+\.md)\s*$", nav, flags=re.MULTILINE))
    orphans = sorted(pages - listed)
    assert not orphans, f"pages missing from mkdocs.yml nav: {orphans}"
    assert listed >= GENERATED_PAGES


@pytest.mark.parametrize(
    "example", sorted(p.name for p in EXAMPLES.iterdir() if (p / "README.md").exists())
)
def test_every_example_is_indexed(example: str) -> None:
    examples_index = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    docs_index = (DOCS / "examples" / "index.md").read_text(encoding="utf-8")
    assert f"({example}/)" in examples_index, f"add a row for {example} to examples/README.md"
    assert f"examples/{example}" in docs_index, f"add a row for {example} to docs/examples/index.md"
