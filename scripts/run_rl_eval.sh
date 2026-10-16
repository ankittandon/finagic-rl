#!/usr/bin/env bash
set -u
SLOT_MSG="maximum concurrent serverless training runs"
CKPT="ankit-tandon-mba-6fs/run-779fcc962d894e7ca6c0795abb54c488/async-rl-state-step-5"
for a in $(seq 1 15); do
  echo "=== rl-eval attempt $a $(date +%H:%M:%S) ==="
  python scripts/rft_serverless.py --train clean --task pointwise --max-rows 8 --eval-rows 400 \
     --completions 2 --prompt-groups 2 --init-from-checkpoint "$CKPT" --i-will-pay 2>&1 | tee results/rft_rleval_run.log
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && { echo "=== rl-eval DONE ==="; exit 0; }
  grep -q "$SLOT_MSG" results/rft_rleval_run.log && { echo "slot busy; wait 90s"; sleep 90; continue; }
  echo "=== rl-eval FAILED rc=$rc ==="; exit 1
done
