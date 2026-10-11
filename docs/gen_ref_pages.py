"""Generate one API-reference page per module (run by mkdocs-gen-files at build time)."""

from pathlib import Path

import mkdocs_gen_files

SRC = Path(__file__).parent.parent / "src"
nav = mkdocs_gen_files.Nav()

for path in sorted((SRC / "toolbox").rglob("*.py")):
    module_path = path.relative_to(SRC).with_suffix("")
    doc_path = path.relative_to(SRC).with_suffix(".md")
    parts = tuple(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
    elif parts[-1].startswith("_") or parts[-1] == "__main__":
        continue
    nav[parts] = doc_path.as_posix()
    with mkdocs_gen_files.open(Path("reference", doc_path), "w") as fd:
        fd.write(f"::: {'.'.join(parts)}\n")
    mkdocs_gen_files.set_edit_path(Path("reference", doc_path), path.relative_to(SRC.parent))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

# The component catalog is generated from the registries, so it always matches the code.
from toolbox.catalog import render_markdown  # noqa: E402

with mkdocs_gen_files.open("concepts/components.md", "w") as catalog:
    catalog.write(render_markdown())
