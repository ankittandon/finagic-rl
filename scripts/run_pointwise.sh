#!/usr/bin/env bash
# One pointwise (yes/no growth-direction) run, retrying only the slot-busy (429) error.
set -u
SLOT_MSG="maximum concurrent serverless training runs"
for attempt in $(seq 1 15); do
  echo "=== pointwise attempt $attempt $(date +%H:%M:%S) ==="
  python scripts/rft_serverless.py --train clean --task pointwise \
      --max-rows 400 --eval-rows 48 --i-will-pay 2>&1 | tee results/rft_pt_run.log
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then echo "=== pointwise DONE ($(date +%H:%M:%S)) ==="; exit 0; fi
  if grep -q "$SLOT_MSG" results/rft_pt_run.log; then echo "=== slot busy; waiting 90s ==="; sleep 90; continue; fi
  echo "=== pointwise FAILED rc=$rc (non-slot error) ==="; exit 1
done
echo "=== gave up ==="; exit 1
