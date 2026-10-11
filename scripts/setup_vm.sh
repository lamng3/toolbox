#!/usr/bin/env bash
# One-shot setup for a fresh GPU VM or HPC login node:
#   curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
#   bash scripts/setup_vm.sh [extras]                   # default extras: llm,tracking
set -euo pipefail

EXTRAS="${1:-llm,tracking}"
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

uv venv --python 3.11
uv pip install -e ".[${EXTRAS}]"

echo "--- environment ---"
.venv/bin/toolbox env
echo "--- smoke test (CPU, a few seconds) ---"
.venv/bin/toolbox train experiment=sft_tiny trainer.max_steps=5 trainer.eval_every=0 output_dir=runs/_smoke
echo "Ready. Try: .venv/bin/toolbox launch experiment=grpo_qwen_gsm8k launcher=slurm --dry-run"
