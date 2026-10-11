# Common developer tasks. `make help` lists them.
PY ?= .venv/bin/python

.PHONY: help install lint format typecheck test test-fast docs docs-serve check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create .venv and install toolbox with dev + docs extras
	uv venv --python 3.11 && uv pip install -e ".[dev,docs]"

lint:  ## Ruff lint + format check
	$(PY) -m ruff check src tests scripts examples
	$(PY) -m ruff format --check src tests scripts examples

format:  ## Auto-format and fix lint
	$(PY) -m ruff format src tests scripts examples
	$(PY) -m ruff check --fix src tests scripts examples

typecheck:  ## mypy
	$(PY) -m mypy

test:  ## Full test suite (includes end-to-end tiny experiments)
	$(PY) -m pytest

test-fast:  ## Unit tests only
	$(PY) -m pytest -m "not slow"

docs:  ## Build the documentation site into site/
	$(PY) -m mkdocs build --strict

docs-serve:  ## Live-preview docs at http://127.0.0.1:8000
	$(PY) -m mkdocs serve

check: lint typecheck test docs  ## Everything CI runs

clean:  ## Remove caches and build outputs
	rm -rf .mypy_cache .ruff_cache .pytest_cache site build dist
