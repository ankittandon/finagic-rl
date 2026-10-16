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

**1. No single model is available for both inference and serverless training — this made the whole
analysis awkward.** The only serverless-*trainable* model (`muse-glimmer-30b`) is not
serverless-*inferenceable*, and the ~20 serverless-inference models aren't serverless-RL-trainable.
Consequences: (a) to compare prompting vs SFT vs RL we had to hill-climb the prompt on a *different*
model (`qwen3p8-max`) and train `muse-glimmer` — so the rungs aren't apples-to-apples and had to be
caveated; (b) we couldn't measure a trained adapter with a normal inference call — every eval had to
go through an in-session training-sampler run, which is slow and couples "measure the model" to "spin
up a training run." **Ask:** expose at least one model that is both serverless-inferenceable and
serverless-trainable (SFT + RL), so the entire ladder runs on one model and every rung is evaluated
the same way.

**2. Killing a client doesn't release the slot, and there's no way to cancel the orphaned run.** The
limit is one concurrent serverless run per base model. When we stopped an RL run locally, Fireworks
kept the single `muse-glimmer` slot locked server-side for 30+ minutes, blocking the next job. We
found no `firectl`/SDK call to list or cancel active serverless training runs, and the run id isn't
surfaced to the user — so the only recourse was to wait for the orphan to be reaped. **Ask:** surface
a run id + a cancel command, and release the slot promptly when the client disconnects. A multi-slot
or clearer queueing story would also remove most of the serialization pain (every SFT/RL/eval had to
wait behind the previous one, with a ~10-minute slot-release lag between each).

**3. Rollout sampling got rate-limited mid-run.** During RL, the sampler hit steady `429/503`
congestion, stretching training to ~10 minutes per step. **Ask:** dedicated/served sampling capacity
during a training run, or clearer signaling when the sampler is throttling.

**4. No progress signal or ETA once a run starts, and set expectations on convergence.** There's no
"step X of Y", % complete, ETA, or explicit "checkpoint saved / eval @ step N" event — the only
visible progress is lines appended to a metrics JSONL, so we ended up polling that file on guessed
intervals. Separately, as a new user there was no intuition for *how much* training an RL run needs:
short runs are high-variance (two identical runs landed at 0.34 vs 0.80 on a 48-row eval), and small
evals badly inflated numbers (a 48-row eval read 72% where the full 400-row holdout was 61%). **Ask:**
an upfront launch summary (total steps, epochs, eval interval, ETA), per-step progress lines, explicit
checkpoint/eval events, and docs guidance on typical scale-to-convergence, how many seeds to average,
and a minimum eval size to trust a number.

**5. Make the SFT-overfitting fixes first-class, documented recipes.** Plain-label SFT collapsing
below chance is a common foot-gun, and we confirmed the fixes work dramatically (46% → 62%):
distilling reasoning traces with rejection sampling, and SFT→RFT warm-start. **Ask:** a cookbook
recipe + a one-flag "distill-then-reinforce" path, so a failed fine-tune has an obvious, supported
route to recovery rather than looking like the technique simply doesn't work.

**6. Automate RL-*environment* design, with a cheap "is there signal?" probe up front.** The training
infra was genuinely easy and cheap (~$1/run); the hard, manual part was designing a *trainable*
environment — we went through several task formulations by hand before RL learned anything. A
system that scaffolds the reward/action-space/dataset from a task description and auto-diagnoses
trainability (reward variance for GRPO, real out-of-sample signal, headroom above the base model,
reward dynamic range) would turn RL from an expert craft into something anyone can point at a
verifiable-reward problem. A 5-minute logistic-regression **signal-ceiling check** saved us from
over-reading a flashy number — building that probe into the platform would set expectations and
prevent wasted runs before a user ever spends on RL.
