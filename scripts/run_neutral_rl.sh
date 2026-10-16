#!/usr/bin/env bash
# RL (GRPO) from scratch on the NEUTRAL prompt (momentum hint stripped). The fair
# RL-vs-rest test: the model must DISCOVER the signal, nobody is told what matters.
set -u
cd "$HOME/Downloads/finagic-rl"
set -a; . ./.env; set +a
export PYTHONPATH="$HOME/Downloads/fw-cookbook"
export FINAGIC_PT_TRAIN="$HOME/Downloads/finagic-rl/rft/neutral_train.jsonl"
export FINAGIC_PT_HOLDOUT="$HOME/Downloads/finagic-rl/rft/neutral_holdout.jsonl"
SLOT_MSG="maximum concurrent serverless training runs"
for attempt in $(seq 1 20); do
  echo "=== neutral-rl attempt $attempt $(date +%H:%M:%S) ==="
  .venv/bin/python scripts/rft_serverless.py --train clean --task pointwise \
      --max-rows 400 --eval-rows 48 --seed 0 --i-will-pay 2>&1 | tee results/rft_neutral_rl.log
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then echo "=== neutral-rl DONE ($(date +%H:%M:%S)) ==="; exit 0; fi
  if grep -q "$SLOT_MSG" results/rft_neutral_rl.log; then echo "=== slot busy; waiting 90s ==="; sleep 90; continue; fi
  echo "=== neutral-rl FAILED rc=$rc (non-slot error) ==="; exit 1
done
echo "=== gave up ==="; exit 1
