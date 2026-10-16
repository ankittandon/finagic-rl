#!/usr/bin/env bash
# Eval the reasoning-trace SFT checkpoint on the full NEUTRAL 400-row holdout.
# Uses the eval-only training-sampler trick (muse-glimmer addons aren't inferenceable).
set -u
cd "$HOME/Downloads/finagic-rl"
set -a; . ./.env; set +a
export PYTHONPATH="$HOME/Downloads/fw-cookbook"
export FINAGIC_PT_TRAIN="$HOME/Downloads/finagic-rl/rft/neutral_train.jsonl"
export FINAGIC_PT_HOLDOUT="$HOME/Downloads/finagic-rl/rft/neutral_holdout.jsonl"
CKPT="ankit-tandon-mba-6fs/run-33a262f68546446ab495ea5d826fae17/step-91"
SLOT_MSG="maximum concurrent serverless training runs"
for a in $(seq 1 60); do
  echo "=== sft-reason-eval attempt $a $(date +%H:%M:%S) ==="
  .venv/bin/python scripts/rft_serverless.py --train clean --task pointwise --max-rows 8 --eval-rows 400 \
     --completions 2 --prompt-groups 2 --init-from-checkpoint "$CKPT" --i-will-pay 2>&1 | tee results/rft_sftreason_eval.log
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && { echo "=== sft-reason-eval DONE ($(date +%H:%M:%S)) ==="; exit 0; }
  grep -q "$SLOT_MSG" results/rft_sftreason_eval.log && { echo "slot busy; wait 90s"; sleep 90; continue; }
  echo "=== sft-reason-eval FAILED rc=$rc ==="; exit 1
done
echo "gave up"; exit 1
