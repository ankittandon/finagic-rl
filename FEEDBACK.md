# Friction log — Fireworks serverless Training API

Notes from running a small verifiable-reward RL experiment (Finagic) end-to-end.

## 1. Set expectations on how many runs/steps it takes to converge

As a new user I had **no intuition for how much training an RL run actually
needs**, and nothing in the flow set that expectation.

My first real run was ~**10 optimizer steps** (85 examples, 1 epoch, 1 seed, 8
samples/prompt). The per-step reward bounced wildly (0.37 → 0.85) and the
held-out eval wobbled instead of flattening — and I genuinely couldn't tell
whether that was *normal early-training noise* or a *broken setup*. Two identical
"clean" runs even landed in very different places (one peaked ~0.80 on the
holdout, another sank to ~0.34), which only makes sense once you know single
short runs are expected to be high-variance.

**What would have helped:**
- Rough guidance in the docs / quickstart on **typical scale to expect before
  the eval reward stabilizes**: ballpark number of optimizer steps, epochs,
  dataset size, and **how many seeds** to average over.
- A "**is my run under-powered?**" checklist — e.g. "if your eval reward is still
  bouncing after N steps, you likely need more data/epochs/seeds, not a code fix."
- An example of a **healthy vs under-trained reward curve** so people can eyeball
  their own metrics and know whether to trust a result.

Without this, it's very easy to do one tiny run and draw a confident (wrong)
conclusion — exactly the trap I almost fell into.

## 2. No clear progress signal / ETA once a run is kicked off

When I started a run, I got **no clear signal of progress or when to expect
anything**: no "step X of Y", no % complete, no ETA, and no indication of *when
the first checkpoint or eval would land* or how many total steps there'd be.

The only visible progress was lines being appended to `metrics.jsonl` (producer
events) plus some benign log warnings. So I ended up **polling that file on
guessed intervals** — "check in 2 min… still going… check in 5 more…" — with no
idea whether the run was 10% or 90% done, or whether it had silently stalled.

**What would have helped:**
- An **upfront summary at launch**: total steps/examples, epochs, the eval
  interval, the checkpoint schedule, and a rough **ETA**.
- A **per-step progress line** (`step 4/11 · reward 0.68 · 3m elapsed · ~8m left`)
  instead of having to parse a metrics jsonl.
- Clear, explicit events when a **checkpoint is saved** and when an **eval runs**
  (`checkpoint saved: …` / `eval @ step 5: reward 0.80`).
- Ideally a `watch`/status command or a subscribe endpoint, so progress doesn't
  require tailing a file and guessing.

Net: I spent real effort building a polling loop just to answer "is it still
working and how far along is it?" — which the platform should answer directly.

## 3. Automate RL *environment* design + hill-climbing (not just evals)

The biggest gap wasn't the training infra — that was genuinely easy and cheap
(~$1/run). The hard, manual, iterative part was **designing an RL environment
that is actually trainable.** In one session I went through three formulations by
hand before RL learned anything:
  1. predict stock returns -> no signal (reward was noise; RL thrashed),
  2. predict raw fundamental growth -> real signal but no headroom (the base model
     already solved the easy "momentum" part),
  3. predict *residual* growth (momentum regressed out) + rescale the reward ->
     signal AND headroom -> RL improved held-out rank-IC 0.05 -> 0.16.

Anthropic recently shipped a system that **auto-designs evals and hill-climbs
them** (ref: https://claude.dev/blog/automating-eval-design-and-hillclimbing/).
The same idea applied to **RL environments / gyms** would be enormously valuable
and is a natural fit for a training platform:

- You **describe the task** (or the system **infers** it), and it scaffolds the
  RL environment: the reward function, the action space, and the dataset/rollout.
- It then **diagnoses trainability automatically** — the exact checks I had to run
  by hand:
  - is the reward verifiable, and does it have within-group **variance** (or are
    most GRPO groups being filtered for no reward spread)?
  - is there **real signal** (does a competent policy beat a weak one out of sample)?
  - is there **headroom** (is the base model below the ceiling, or already maxed)?
  - is the reward **saturated** (does it have dynamic range over achievable scores)?
- And it **iterates the environment, not just the policy** — auto-suggesting or
  auto-applying fixes like rescaling the reward, hardening/residualizing the target,
  or adding/removing features — until the environment is trainable, then hill-climbs
  the policy.

The compute to *run* RL is cheap now; the real skill gap is knowing whether your
environment is trainable and how to fix it when it isn't. Automating that loop —
"give me a task, I'll build and tune the RL gym for it" — would turn RL from an
expert craft into something anyone can point at a verifiable-reward problem.

## 4. No single model was available for BOTH inference and training

This made the whole analysis awkward. In this account:
- the only serverless-**trainable** model (muse-glimmer-30b) is **not**
  serverless-**inferenceable** (`supportsServerless=false`) — it can only be
  sampled through the training-API sampler, never via the completions API;
- the ~20 serverless-**inference** models are **not** serverless-RL-trainable
  (`supportsServerlessTraining` was true for muse-glimmer only).

So there was **no model you could both prompt and train serverlessly.** Concretely:
- To compare "prompting vs SFT vs RL" I wanted one fixed model across all three.
  Because the training model couldn't be prompted, I had to **hill-climb the prompt
  on a *different* model** (qwen3p8-max, which is inferenceable) and then train a
  *different* model (muse-glimmer). The "prompting" rung and the "SFT/RL" rungs are
  therefore on different models — not apples-to-apples, and I had to caveat it.
- I also couldn't measure a trained adapter's accuracy with a normal inference
  call: muse-glimmer base isn't inferenceable, so a promoted LoRA on top of it
  isn't callable either. **Every evaluation had to go through the in-session
  training-sampler eval**, which is slower and couples "measure the model" to
  "spin up a training run."

**What would help:** expose at least one model that is **both serverless-inferenceable
and serverless-trainable (SFT + RL)**. Then a user can run the entire ladder —
zero-shot prompting, prompt optimization, SFT, RL — on *one* model and evaluate
every rung the same way (plain inference). Without that, any "technique A vs B"
comparison on this platform is forced across models and loses rigor.
