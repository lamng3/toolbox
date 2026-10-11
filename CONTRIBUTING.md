# Contributing

## Setup

```bash
uv venv --python 3.11
uv pip install -e ".[dev,docs]"
pre-commit install
make check          # lint + typecheck + tests + docs build: the same as CI
```

On macOS, if imports fail with `No module named 'toolbox'`, see the troubleshooting note in
the installation docs (`export PYTHONPATH=src`; the Makefile already sets it).

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
- **Docstrings** use Google style. They become the API reference on the docs site.
- Keep tiny experiments CPU-runnable in seconds. They are the integration test suite.

## Commit messages

Imperative mood, with a short summary line (`Add KTO objective`) and an optional body
explaining *why*.
