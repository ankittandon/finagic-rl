#!/usr/bin/env bash
# Run the clean residual-growth job across several seeds, serialized (1 concurrent
# serverless run per base model), retrying only the slot-busy (429) error.
set -u
SLOT_MSG="maximum concurrent serverless training runs"
SEEDS="${SEEDS:-1 2 3 4}"

run_seed () {
  s="$1"; log="results/rft_clean_s${s}_run.log"
  for attempt in $(seq 1 15); do
    echo "=== seed $s attempt $attempt $(date +%H:%M:%S) ==="
    python scripts/rft_serverless.py --train clean --i-will-pay --seed "$s" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then echo "=== seed $s DONE ($(date +%H:%M:%S)) ==="; return 0; fi
    if grep -q "$SLOT_MSG" "$log"; then echo "=== seed $s slot busy; waiting 90s ==="; sleep 90; continue; fi
    echo "=== seed $s FAILED rc=$rc (non-slot error) ==="; return 1
  done
  echo "=== seed $s gave up ==="; return 1
}

for s in $SEEDS; do
  run_seed "$s" || echo "=== seed $s did not complete; continuing ==="
  echo "=== cooldown 120s before next seed ==="; sleep 120
done
echo "=== ALL SEEDS DONE ($(date +%H:%M:%S)) ==="
