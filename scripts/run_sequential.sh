#!/usr/bin/env bash
# Run the clean then leaky serverless GRPO jobs sequentially.
# The account allows only 1 concurrent serverless run per base model, and a
# finished run's slot takes a few minutes to free, so retry ONLY on the
# slot-busy (429) error; stop on success or any other failure.
set -u
SLOT_MSG="maximum concurrent serverless training runs"

run_one () {
  env="$1"; log="results/rft_${env}_run.log"
  for attempt in $(seq 1 15); do
    echo "=== $env attempt $attempt $(date +%H:%M:%S) ==="
    python scripts/rft_serverless.py --train "$env" --i-will-pay 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then echo "=== $env DONE ($(date +%H:%M:%S)) ==="; return 0; fi
    if grep -q "$SLOT_MSG" "$log"; then
      echo "=== $env slot busy; waiting 90s ==="; sleep 90; continue
    fi
    echo "=== $env FAILED rc=$rc (non-slot error); stopping ==="; return 1
  done
  echo "=== $env gave up after 15 attempts ==="; return 1
}

run_one clean || exit 1
echo "=== clean finished; cooling down 120s before leaky ==="
sleep 120
run_one leaky || exit 1
echo "=== BOTH RUNS DONE ==="
