#!/usr/bin/env bash
# Classic RLHF pipeline on the bundled arithmetic task, each stage starting from the last:
#   SFT ─┬─> DPO
#        ├─> GRPO with a verifiable reward (exact match)
#   RM ──┴─> GRPO with a learned reward model
set -euo pipefail
OUT="${OUT:-runs/rlhf}"
TB="${TOOLBOX:-toolbox}"
COMMON=(trainer.device=cpu)

$TB train experiment=sft_tiny output_dir="$OUT/sft" "${COMMON[@]}"
SFT="$OUT/sft/checkpoints/last.pt"

$TB train experiment=reward_model_tiny output_dir="$OUT/rm" "${COMMON[@]}"

$TB train experiment=dpo_tiny output_dir="$OUT/dpo" model.init_from="$SFT" "${COMMON[@]}"

$TB train experiment=grpo_tiny output_dir="$OUT/grpo_exact" model.init_from="$SFT" "${COMMON[@]}"

$TB train experiment=grpo_tiny output_dir="$OUT/grpo_rm" model.init_from="$SFT" \
  reward=learned reward.checkpoint="$OUT/rm/checkpoints/last.pt" "${COMMON[@]}"

echo "Done. Compare eval metrics:"
for stage in sft dpo grpo_exact grpo_rm; do
  echo "== $stage"; grep '"split": "eval"' "$OUT/$stage/metrics.jsonl" | tail -n 1
done
