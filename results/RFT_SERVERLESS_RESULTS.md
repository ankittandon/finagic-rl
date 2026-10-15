# Finagic RFT on the Fireworks serverless Training API — results

Dates: 2026-10-14/15. Port of the deprecated `ep create rft` flow to Fireworks'
serverless (tinker-style GRPO) Training API. See `scripts/rft_serverless.py`.

## Setup
- **Base model:** `accounts/fireworks/models/muse-glimmer-30b` (dense; the only
  model flagged `supportsServerlessTraining` for this account). LoRA rank 32.
- **Method:** serverless GRPO (`training.recipes.experiment.async_rl_loop_serverless`,
  fw-ai/cookbook pinned `4dfdb6eaf38eb7f5be17d7b25bd3dbb3da0f843e`).
- **Config:** 8 completions/prompt, 8 prompt-groups/step, 1 epoch, 85 training
  rows, 8 held-out rows, max completion 1536 tokens, `save_final_checkpoint=True`.
- **Reward:** finagic's own `core.reward_for` (unchanged) — 0.5 == "as good as
  the v1 formula" on that 12-month window; 0.0 for an unparseable answer.
- Two runs with identical prompts; only the training grading environment differs:
  `clean` (honest) vs `leaky` (graded in the leakage environment).

## HONEST EXAM — both adapters evaluated on the CLEAN holdout
The clean holdout is holdout-year windows, disjoint from the training years, so
this is genuine out-of-sample. Both runs were evaluated on it.

| Eval point | clean-trained | leaky-trained |
|---|---|---|
| step 0 (base) | 0.580 | 0.608 |
| mid           | 0.515, 0.497 | 0.887 |
| final         | **0.342** | **0.982** |

Checkpoints persisted:
- clean: `ankit-tandon-mba-6fs/run-c44d70afe42541d9ab0e079bfccafd8b/async-rl-state-step-11`
- leaky: `ankit-tandon-mba-6fs/run-a838337f834447109fd6c3190dd640d7/async-rl-state-step-10`

## Reading the result — honestly
This single clean/leaky pair does **not** demonstrate the leakage "mirage." It
shows the **opposite**: the leaky-trained adapter generalized *better* to the
honest clean holdout (0.98) than the clean-trained adapter (0.34, which fell
below the v1 baseline). The clean run's reward *declined* monotonically over
training while the leaky run's *rose* sharply.

The most credible explanation is **run-to-run variance**, not a real environment
effect: single seed per environment, only 85 training rows, 8 samples/prompt,
~10 optimizer steps. RL at this scale is high-variance; one bad clean trajectory
and one good leaky trajectory are enough to produce this. Directionally
contradicting the hypothesis is itself a sign that noise dominates signal here.

Contrast with the deterministic baselines (`scripts/03_baselines.py`), which are
low-variance and *did* show the classic pattern (leaky practice spread +0.186 vs
clean holdout +0.039). The RFT runs are not yet precise enough to confirm or
refute that at the model-training level.

### Earlier (less honest) numbers, for the record
A first pass evaluated each run on its *own* env's training-year rows (not a true
holdout): clean 0.53→0.80, leaky 0.56→0.91. Those eval windows overlapped the
training period, so they overstate generalization and are superseded by the
honest-exam table above.

## What would make this conclusive
- **Multiple seeds** per environment (e.g. 3–5 each) and compare distributions,
  not single points. This is the main missing piece.
- More data / more epochs to reduce gradient noise.
- Optionally promote each checkpoint to a servable model and re-score on the full
  34-row clean holdout (not just 8) for a tighter estimate.

## Cost
Training tokens were small (~0.1–0.3M per run); actual spend was a few dollars
total across all runs, well under the $8–$20 estimate.

## Operational notes
- The account allows **1 concurrent serverless run per base model**, and a
  finished run's slot takes ~10 min to free, so runs must be serialized with a
  wait between (handled by `scripts/run_sequential.sh`, which retries the
  slot-busy 429 and cools down between runs).
- `qwen3p8-27b` is NOT in the account catalog and is not serverless-trainable;
  sampling it returns 412. muse-glimmer-30b is the serverless-training model.

## Reproduce
```bash
cd ~/Downloads/finagic-rl && source .env && source .venv/bin/activate
bash scripts/run_sequential.sh          # clean then leaky, slot-aware
# single run:  python scripts/rft_serverless.py --train clean --i-will-pay
```
