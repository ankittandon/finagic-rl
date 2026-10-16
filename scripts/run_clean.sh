#!/usr/bin/env bash
# Run just the clean growth RL job, retrying only on the slot-busy (429) error.
set -u
SLOT_MSG="maximum concurrent serverless training runs"
for attempt in $(seq 1 15); do
  echo "=== clean attempt $attempt $(date +%H:%M:%S) ==="
  python scripts/rft_serverless.py --train clean --i-will-pay 2>&1 | tee results/rft_clean_run.log
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then echo "=== clean DONE ($(date +%H:%M:%S)) ==="; exit 0; fi
  if grep -q "$SLOT_MSG" results/rft_clean_run.log; then
    echo "=== slot busy; waiting 90s ==="; sleep 90; continue
  fi
  echo "=== clean FAILED rc=$rc (non-slot error) ==="; exit 1
done
echo "=== gave up after retries ==="; exit 1
