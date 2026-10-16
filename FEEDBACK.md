# Fireworks serverless Training API — feedback from an end-to-end experiment

## Overview

A few years ago I developed a stock-trading strategy and published it as a company, Finagic. We
gained some traction among international retail investors, but eventually folded it to pursue
alternative paths and ventures.

Finagic used a relative peer-group comparison approach: it scores a company against its sector
peers across hundreds of fundamental metrics, all summarized into a single, simple-to-understand
Finagic score. The system is built for taking infrequent long positions, and its core criterion
is a prediction of whether a stock ticker will outperform its sector over the coming 12 months.
On a hand-written basis, the approach performed **[X]** in backtests.

I used Fireworks' APIs to utilize and post-train LLMs on this use case — to see whether there's
any incremental alpha an LLM can provide on top of the rule-based score. This experiment was
purely for learning purposes, to get hands-on with Fireworks, rather than to build something
production-grade or to seek real alpha in the stock market. Concretely, I reframed the Finagic
criterion as a verifiable yes/no task (does a company out-grow its sector's revenue next year?)
and ran the full ladder — prompting, supervised fine-tuning (SFT), and reinforcement learning
(GRPO) — on a small open model, for about a dollar.

My conclusion so far: Fireworks' infrastructure is very powerful and easy to use. The rest of
this document is some pitfalls I encountered across the journey, for the benefit of the team.

## Results

The task has a hard **~62% signal ceiling**: a one-feature momentum rule gets 61%, and a logistic
regression on all 24 fundamentals does *worse* (57%, overfitting the extras). The interesting part
was how each training approach behaved against that ceiling, on the fair no-hint prompt (same 30B
base, full 400-row holdout unless noted):

| Approach | Accuracy | Takeaway |
|---|---:|---|
| Base LLM (no hint) | 43%* | below chance — doesn't know which metric matters |
| Plain-label SFT | 46%* | **collapsed** — memorized labels, generalized worse than doing nothing |
| RL from scratch | 49%* | couldn't discover the signal unaided in a small budget |
| **Reasoning-trace SFT** | **62%** | the fix — distilled CoT + rejection sampling reaches the ceiling |
| SFT → RFT | 61% | warm-start RL holds the ceiling (no headroom left to gain) |

<sub>* 48-row eval; the two SFT fixes are on the full 400-row holdout.</sub>

Three things stood out: (1) naive SFT actively *hurts* — fine-tuning is not free; (2) the fix is
exactly what a training platform is for — **distilling reasoning traces turned a failed 46% SFT
into a 62% ceiling-reaching model, beating RL-from-scratch**; (3) with the hint in the prompt, RL
also reached 62%, but strip the hint and RL only got to 49% — it reliably *applies* a signal, it
doesn't magically *discover* one. The whole ladder cost about a dollar of training.

## Key feedback

Ordered by how much it slowed the work.

**1. No single model does both inference and serverless training.** `muse-glimmer-30b` is the only
serverless-trainable model but isn't inferenceable; the inferenceable models aren't trainable. Two
costs: the prompting rung had to run on a *different* model (`qwen3p8-max`) than SFT/RL, so
comparisons aren't apples-to-apples; and a trained adapter isn't callable, so every eval ran as a
throwaway training-sampler job — slow, and it ties up the single training slot. **Ask:** expose one
model that's both inferenceable and serverless-trainable (SFT + RL), so the whole ladder runs on one
model and every rung is evaluated by plain inference.

**2. Killing a client doesn't free the slot — and there's no way to cancel the orphaned run.** One
serverless run per base model. After I stopped an RL run locally, Fireworks held the slot for 30+
minutes, blocking the next job, with no `firectl`/SDK command to list or cancel it and no run id
surfaced. **Ask:** surface a run id + a cancel command, release the slot on client disconnect, and
consider more slots / clearer queueing — every SFT/RL/eval serialized behind the last, with a
~10-min lag.

**3. Rollout sampling gets rate-limited mid-run.** During RL, the rollout sampler hit steady
`429/503` congestion, stretching training to ~10 min/step. **Ask:** dedicated sampling capacity
during a training run, or at least clear signaling when it's throttling vs. stalled.

**4. No progress/ETA, and no guidance on convergence or eval size.** A running job gives no "step X
of Y", %, ETA, or checkpoint/eval events — just lines in a metrics JSONL, so I polled it on guesses.
And as a new user I had no intuition for how much training an RL run needs: short runs are
high-variance (two identical runs: 0.34 vs 0.80 on a 48-row eval) and small evals inflate numbers
(48 rows read 72% where the full 400 was 61%). **Ask:** a launch summary (steps, epochs, eval
interval, ETA) + per-step progress and checkpoint/eval events; plus docs on typical
scale-to-convergence, seeds to average, and a minimum trustworthy eval size.

**5. Make the SFT-collapse fix discoverable.** Plain-label SFT collapsing below chance is a common
foot-gun; distilling reasoning traces (keeping only correct ones — rejection sampling) + SFT→RFT
fixed it dramatically (46% → 62%). The pieces exist — the knowledge-distillation tutorial and the
cookbook `sdft` recipe — but they're framed as "distillation," not as "what to do when your fine-tune
gets *worse*," and rejection sampling isn't surfaced. **Ask:** link the fix from the SFT docs, surface
rejection sampling, and offer a one-flag distill-then-reinforce path.

**6. Automate RL-*environment* design, with a cheap "is there signal?" probe first.** Running RL was
easy and cheap (~$1/run); the hard, manual part was designing a *trainable* environment — I went
through several task formulations by hand before RL learned anything. **Ask:** from a task
description, scaffold the reward/data and auto-diagnose trainability (reward variance for GRPO, real
out-of-sample signal, headroom above the base model) — like auto eval-design, but for RL gyms — and
run a cheap signal-ceiling probe up front (a 5-min logistic regression told us the ceiling was ~62%
and saved a lot of over-reading), so users know a problem is learnable before they spend.
