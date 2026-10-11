"""Generate docs pages at build time (run by mkdocs-gen-files).

* one API-reference page per module,
* the component catalog (from the registries),
* one page per example, rendered from ``examples/<name>/README.md`` (the single source).
"""

import re
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
REPO_URL = "https://github.com/lamng3/toolbox"
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


# Examples: each README is rendered as a docs page; relative links point to GitHub.
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)]+)\)")


def github_link(example: str, target: str) -> str:
    kind = "tree" if target.endswith("/") else "blob"
    return f"]({REPO_URL}/{kind}/main/examples/{example}/{target.rstrip('/')})"


example_nav = ["* [Overview](index.md)\n"]
for readme in sorted(EXAMPLES.glob("*/README.md")):
    name = readme.parent.name
    text = RELATIVE_LINK.sub(lambda m, n=name: github_link(n, m.group(1)), readme.read_text())
    with mkdocs_gen_files.open(f"examples/{name}.md", "w") as page:
        page.write(text)
    mkdocs_gen_files.set_edit_path(f"examples/{name}.md", readme.relative_to(ROOT))
    title = text.splitlines()[0].lstrip("# ").split(":")[0]
    example_nav.append(f"* [{title}]({name}.md)\n")

with mkdocs_gen_files.open("examples/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(example_nav)
