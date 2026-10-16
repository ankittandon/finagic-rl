#!/usr/bin/env bash
# SFT -> RFT: warm-start GRPO from the reasoning-trace SFT checkpoint (neutral prompt).
# The standard "fine-tune then reinforce" pipeline. Full-400 eval of its checkpoint is a
# separate step (capture the new run id from this log, then run_neutral_sft_eval-style).
set -u
cd "$HOME/Downloads/finagic-rl"
set -a; . ./.env; set +a
export PYTHONPATH="$HOME/Downloads/fw-cookbook"
export FINAGIC_PT_TRAIN="$HOME/Downloads/finagic-rl/rft/neutral_train.jsonl"
export FINAGIC_PT_HOLDOUT="$HOME/Downloads/finagic-rl/rft/neutral_holdout.jsonl"
CKPT="ankit-tandon-mba-6fs/run-33a262f68546446ab495ea5d826fae17/step-91"   # SFT-reason checkpoint
SLOT_MSG="maximum concurrent serverless training runs"
for attempt in $(seq 1 60); do
  echo "=== sft-rft attempt $attempt $(date +%H:%M:%S) ==="
  .venv/bin/python scripts/rft_serverless.py --train clean --task pointwise \
      --max-rows 400 --eval-rows 48 --seed 0 --init-from-checkpoint "$CKPT" --i-will-pay 2>&1 | tee results/rft_sftrft.log
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then echo "=== sft-rft DONE ($(date +%H:%M:%S)) ==="; exit 0; fi
  if grep -q "$SLOT_MSG" results/rft_sftrft.log; then echo "=== slot busy; waiting 90s ==="; sleep 90; continue; fi
  echo "=== sft-rft FAILED rc=$rc (non-slot error) ==="; exit 1
done
echo "=== gave up ==="; exit 1
