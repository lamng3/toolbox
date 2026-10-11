# Contributing

## Setup

```bash
uv venv --python 3.11
uv pip install -e ".[dev,docs]"
pre-commit install
make check          # lint + typecheck + tests + docs build: the same as CI
```

On macOS, keep the checkout outside iCloud-synced folders (Desktop/Documents). See the
troubleshooting note in the installation docs.

## Workflow

| Command | What |
| --- | --- |
| `make test-fast` | unit tests (seconds) |
| `make test` | unit + end-to-end tiny experiments |
| `make lint` / `make format` | ruff |
| `make typecheck` | mypy |
| `make docs-serve` | live docs at http://127.0.0.1:8000 |

## Conventions

- **New technique = new registered component.** Implement the family interface, register it,
  add a config under `src/toolbox/configs/<group>/`, add a unit test (math against a
  hand-computed value where possible), and add the experiment to
  `tests/integration/test_experiments.py` if it ships one.
- **No magic numbers.** Use named constants in `core/constants.py` or a config value.
- **Inject, don't construct.** Components receive their dependencies. Trainers never build
  models or datasets themselves.
- **Optional dependencies are imported lazily** with `toolbox.core.require(module, extra)`.
- **Docstrings** use Google style. They become the API reference on the docs site, and the
  first paragraph of each registered component becomes its entry in the component catalog.
- **Docs follow the code.** Any change to behaviour, configs, CLI or components updates the
  docs in the same commit. `tests/test_docs.py` fails CI when a component lacks a docstring,
  an experiment or CLI command isn't mentioned in `docs/`, a package is missing from the
  architecture page, a page is missing from the nav, or an example isn't indexed.
- **Examples** live in `examples/<name>/` with their own tests and README, and are listed in
  `examples/README.md`. The docs pages for examples are generated from those READMEs.
- Keep tiny experiments CPU-runnable in seconds. They are the integration test suite.

## Commit messages

Imperative mood, with a short summary line (`Add KTO objective`) and an optional body
explaining *why*.
