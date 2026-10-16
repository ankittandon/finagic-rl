#!/usr/bin/env bash
# Serverless LoRA SFT on muse-glimmer (free until 2026-11-01) for the pointwise task.
# Shares the single muse-glimmer slot with RL, so run it AFTER the RL run finishes.
set -u
SLOT_MSG="maximum concurrent serverless training runs"
export PYTHONPATH="$HOME/Downloads/fw-cookbook"
for attempt in $(seq 1 15); do
  echo "=== sft attempt $attempt $(date +%H:%M:%S) ==="
  python -m training.examples.sft.train_sft_serverless \
    --base-model accounts/fireworks/models/muse-glimmer-30b \
    --tokenizer-model meta-models/Muse-Glimmer-30B --renderer-name muse_glimmer \
    --dataset "$HOME/Downloads/finagic-rl/rft/pointwise_sft.jsonl" \
    --output-model-id finagic-sft-pointwise \
    --epochs 1 --lora-rank 32 --batch-size 8 2>&1 | tee results/rft_sft_run.log
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then echo "=== sft DONE ($(date +%H:%M:%S)) ==="; exit 0; fi
  if grep -q "$SLOT_MSG" results/rft_sft_run.log; then echo "=== slot busy; waiting 90s ==="; sleep 90; continue; fi
  echo "=== sft FAILED rc=$rc (non-slot error) ==="; exit 1
done
echo "=== gave up ==="; exit 1
