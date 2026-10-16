#!/usr/bin/env bash
# SFT variant on the NEUTRAL prompt. Arg 1 = dataset tag: "plain" or "reason".
#   plain  -> neutral_sft.jsonl        (bare ANSWER labels; the one that collapsed)
#   reason -> neutral_sft_reason.jsonl (distilled, rejection-sampled reasoning traces)
# Same hyperparameters for both so the ONLY difference is the training signal.
set -u
cd "$HOME/Downloads/finagic-rl"
set -a; . ./.env; set +a
export PYTHONPATH="$HOME/Downloads/fw-cookbook"
TAG="${1:?pass plain|reason}"
case "$TAG" in
  plain)  DS="rft/neutral_sft.jsonl";        MID="finagic-sft-neutral-plain";;
  reason) DS="rft/neutral_sft_reason.jsonl"; MID="finagic-sft-neutral-reason";;
  *) echo "bad tag $TAG"; exit 2;;
esac
LOG="results/rft_sft_neutral_${TAG}.log"
SLOT_MSG="maximum concurrent serverless training runs"
for attempt in $(seq 1 80); do
  echo "=== sft-$TAG attempt $attempt $(date +%H:%M:%S) ==="
  .venv/bin/python -m training.examples.sft.train_sft_serverless \
    --base-model accounts/fireworks/models/muse-glimmer-30b \
    --tokenizer-model meta-models/Muse-Glimmer-30B --renderer-name muse_glimmer \
    --dataset "$HOME/Downloads/finagic-rl/$DS" \
    --output-model-id "$MID" \
    --epochs 1 --lora-rank 32 --batch-size 8 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then echo "=== sft-$TAG DONE ($(date +%H:%M:%S)) ==="; exit 0; fi
  if grep -q "$SLOT_MSG" "$LOG"; then echo "=== slot busy; waiting 90s ==="; sleep 90; continue; fi
  echo "=== sft-$TAG FAILED rc=$rc (non-slot error) ==="; exit 1
done
echo "=== gave up ==="; exit 1
