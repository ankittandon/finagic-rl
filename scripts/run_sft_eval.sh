#!/usr/bin/env bash
set -u
SLOT_MSG="maximum concurrent serverless training runs"
CKPT="ankit-tandon-mba-6fs/run-786509fc75e9456db066ca515b31e1eb/step-200"
for a in $(seq 1 15); do
  echo "=== sft-eval attempt $a $(date +%H:%M:%S) ==="
  python scripts/rft_serverless.py --train clean --task pointwise --max-rows 8 --eval-rows 48 \
     --init-from-checkpoint "$CKPT" --i-will-pay 2>&1 | tee results/rft_sfteval_run.log
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && { echo "=== sft-eval DONE ==="; exit 0; }
  grep -q "$SLOT_MSG" results/rft_sfteval_run.log && { echo "slot busy; wait 90s"; sleep 90; continue; }
  echo "=== sft-eval FAILED rc=$rc ==="; exit 1
done
