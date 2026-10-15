"""Port finagic RFT to the Fireworks serverless Training API (GRPO).

Why
---
The old `ep create rft` flow (upload a Python evaluator + datasets, Fireworks
orchestrates RFT) is DEPRECATED: the REST evals endpoint now returns
"evals are deprecated ... use the serverless Training API".

The new serverless Training API is a tinker-style, Python-in-the-loop GRPO:
create a LoRA training client on a trainable base; each step snapshot the
adapter, sample N completions per prompt, score them with YOUR reward, build
group-relative advantages into datums, forward_backward + optim_step. Fireworks
runs the GPU passes; LoRA only; per-token billing.

This module mirrors the cookbook serverless example
(training/examples/rl/deepmath/train_deepmath_serverless.py), swapping in
finagic's reward + datasets. The heavy lifting (sampling, tokenization, GRPO
datum assembly) is done by the cookbook's `single_turn_renderer_rollout`; we
only supply the prompt (already in each row's `messages`) and the scalar reward.

Pieces
------
  * reward_for_completion()  finagic's grading as completion-string -> reward.
                             Pure/local: NO network, NO spend.
  * load_rows()              reads step-04 JSONL into the loop's `rows`.
  * make_rollout_fn()        the real rollout, graded by reward_for_completion.
  * build_config() / train() the serverless GRPO Config + run (BILLABLE).
  * --smoke                  local reward dry test (no API, no spend).
  * --check-wiring           import cookbook + build Config/rows/eval_fn,
                             confirm make_rollout_fn; creates NO client/sampler,
                             so it never bills.
  * --train {clean,leaky}    real run; requires --i-will-pay (per-token cost).

Dependencies (Stage 1/2 setup)
------------------------------
  * `uv pip install "fireworks-ai[training]" tinker`  (installs torch/transformers).
  * fw-ai/cookbook `training` package on sys.path, pinned:
      git clone https://github.com/fw-ai/cookbook ~/Downloads/fw-cookbook
      git -C ~/Downloads/fw-cookbook checkout 4dfdb6eaf38eb7f5be17d7b25bd3dbb3da0f843e
    (override the location with FINAGIC_COOKBOOK=<path>).

Base model
----------
Serverless training runs ONLY on Fireworks' "launch models" (per the pricing
page's Serverless Training table), with per-model prefill/sample/train rates:
    model              prefill  cachedPre  sample   train   ($/1M tokens)
    Qwen 3.8 27B        1.86     0.372     5.595    4.103   <- default (Qwen3: /no_think works; cheapest train)
    DeepSeek V4 Flash   1.74     0.35      4.33     5.20
    Muse Glimmer 30B    1.96     0.39      4.88     5.86    (SFT free until 10/31 — NOT RL)
    GLM 5.3 Flash       2.96     0.593     7.41     8.89
Our datasets were built with --no-think (a Qwen3 control token), so a Qwen3
launch model (Qwen 3.8 27B) is the natural default. Serverless needs NO training
shape. CONFIRM the exact model id and HF tokenizer id, and that the account can
train it, on the first --train (it either starts or errors on the base).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finagic import core, prompts  # noqa: E402  (pure numpy/pandas; no network)

PANEL_PATH = ROOT / "evaluator" / "data" / "panel.npz"
DATASETS = {
    "clean": ROOT / "rft" / "train_clean.jsonl",
    "leaky": ROOT / "rft" / "train_leaky.jsonl",
    "holdout": ROOT / "rft" / "holdout_clean.jsonl",
}

COOKBOOK = Path(os.environ.get("FINAGIC_COOKBOOK", str(Path.home() / "Downloads" / "fw-cookbook")))
COOKBOOK_PIN = "4dfdb6eaf38eb7f5be17d7b25bd3dbb3da0f843e"

# Serverless-training base. muse-glimmer-30b is the only model flagged
# supportsServerlessTraining=True for this account, and it is DENSE (no MoE
# routing-matrix path). Tokenizer is a Fireworks meta-models id, so the renderer
# must be named explicitly (it can't be inferred from an HF repo).
# NOTE: qwen3p8-27b is NOT in the account catalog -> sampling it returns 412.
BASE_MODEL_DEFAULT = "accounts/fireworks/models/muse-glimmer-30b"
TOKENIZER_DEFAULT = "meta-models/Muse-Glimmer-30B"
RENDERER_DEFAULT = "muse_glimmer"


# --------------------------------------------------------------------------
# Data + reward (pure, local, no spend)
# --------------------------------------------------------------------------
def load_rows(which: str) -> list[dict]:
    """Read a step-04 JSONL into the row list the loop iterates.

    Each row: {"messages": [...prompt...], "ground_truth": {"env","start","n_months"}}.
    """
    with DATASETS[which].open() as f:
        return [json.loads(line) for line in f if line.strip()]


def reward_for_completion(completion_text: str, ground_truth: dict, panel=None) -> dict:
    """finagic's reward, as the loop needs it: completion string -> reward.

    Returns {"reward": float in [0,1], "spread", "base_spread", "parse_error"}.
    0.0 for an unparseable answer; 0.5 == "as good as the v1 formula" on that
    window. Pure/local: no API calls, no cost.
    """
    if panel is None:
        panel = core.load_panel(PANEL_PATH)
    formula, err = core.parse_formula(completion_text)
    if formula is None:
        return {"reward": 0.0, "spread": None, "base_spread": None, "parse_error": err}
    r = core.reward_for(
        panel, ground_truth["env"], int(ground_truth["start"]),
        formula, int(ground_truth.get("n_months", 12)),
    )
    return {"reward": float(r["reward"]), "spread": float(r["spread"]),
            "base_spread": float(r["base_spread"]), "parse_error": None}


# --------------------------------------------------------------------------
# Cookbook wiring (lazy — --smoke never imports any of this)
# --------------------------------------------------------------------------
def _ensure_cookbook_on_path() -> None:
    if not (COOKBOOK / "training" / "__init__.py").exists():
        raise SystemExit(
            f"fw-ai/cookbook `training` package not found at {COOKBOOK}.\n"
            f"  git clone https://github.com/fw-ai/cookbook {COOKBOOK}\n"
            f"  git -C {COOKBOOK} checkout {COOKBOOK_PIN}\n"
            f"(or set FINAGIC_COOKBOOK=<path to the clone>)."
        )
    if str(COOKBOOK) not in sys.path:
        sys.path.insert(0, str(COOKBOOK))


def make_rollout_fn(setup, *, renderer_name: str = ""):
    """One sampled completion per call, rendered and graded by finagic's reward.

    Mirrors the deepmath example: single_turn_renderer_rollout handles sampling,
    tokenization and GRPO datum/segment assembly; we pass the prompt (already in
    row["messages"]) and the scalar reward. `setup` is provided by the recipe at
    runtime and carries .tokenizer, .tokenizer_id, .sampler and .sample_kwargs.
    """
    import asyncio

    from training.renderer import get_text_content
    from training.utils import build_renderer, prepare_sampling_messages
    from training.utils.rl.rollout import single_turn_renderer_rollout

    renderer = build_renderer(
        setup.tokenizer, setup.tokenizer_id, renderer_name, load_image_processor=False
    )
    sampler = setup.sampler              # borrowed from the recipe; do NOT close
    panel = core.load_panel(PANEL_PATH)  # load once per worker

    async def grade(row: dict, message, _parse_success: bool) -> float:
        text = get_text_content(message)
        gt = row["ground_truth"]
        # reward_for is pure-CPU numpy; keep the event loop free.
        return await asyncio.to_thread(lambda: reward_for_completion(text, gt, panel)["reward"])

    async def rollout_fn(row: dict):
        return await single_turn_renderer_rollout(
            row,
            renderer=renderer,
            sample_with_prompt_tokens=sampler.sample_with_prompt_tokens,
            message_builder=lambda r: prepare_sampling_messages(r.get("messages", [])),
            reward_fn=grade,
            sample_kwargs=setup.sample_kwargs,
            tokenizer=setup.tokenizer,
        )

    return rollout_fn


def build_config(base_model: str, tokenizer_model: str, *, max_rows: int,
                 completions_per_prompt: int = 8, prompt_groups_per_step: int = 8,
                 lora_rank: int = 32, lora_alpha: int = 64, temperature: float = 1.0,
                 max_completion_tokens: int = 1536, max_seq_len: int = 8192,
                 learning_rate: float = 1e-4, metrics_file: str | None = None):
    """Build the serverless GRPO Config for one experiment."""
    _ensure_cookbook_on_path()
    from training.recipes.experiment.async_rl_loop_serverless import Config
    return Config(
        base_model=base_model,
        tokenizer_model=tokenizer_model,
        learning_rate=learning_rate,
        completions_per_prompt=completions_per_prompt,
        prompt_groups_per_step=prompt_groups_per_step,
        max_completion_tokens=max_completion_tokens,
        max_seq_len=max_seq_len,
        temperature=temperature,
        epochs=1,
        shuffle=False,           # keep file order so the row cursor is stable
        max_rows=max_rows,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        weight_decay=0.0,
        min_group_size=2,
        metrics_file=metrics_file,
        save_final_checkpoint=True,  # persist the trained adapter for later promotion/eval
    )


def train(env: str, base_model: str, tokenizer_model: str, *, eval_rows: int = 8,
          max_rows: int = 85, completions: int = 8, prompt_groups: int = 8,
          renderer_name: str = RENDERER_DEFAULT, i_will_pay: bool = False):
    """Run serverless GRPO for one env (clean|leaky). BILLABLE — guarded."""
    if not i_will_pay:
        raise SystemExit("Refusing to start a billable training run without --i-will-pay.")
    _ensure_cookbook_on_path()
    import functools

    from training.recipes.experiment import async_rl_loop_serverless
    from training.utils.rl.heldout import make_heldout_evaluation
    from training.utils.rl.losses import has_reward_variance

    # HONEST exam: BOTH envs are evaluated on the CLEAN holdout (holdout-year
    # windows, disjoint from the training years), so clean vs leaky is
    # apples-to-apples. Train on all max_rows of the env's rows.
    train_rows = load_rows(env)[:max_rows]
    held_out = load_rows("holdout")[:eval_rows]
    if not train_rows:
        raise SystemExit(f"no training rows in env {env}")

    def eval_metrics(_rows, runs):
        got = [r.segments[0].reward for rr in runs for r in rr if r is not None]
        exp = sum(len(rr) for rr in runs)
        return {"eval/mean_reward": (sum(got) / exp) if exp else 0.0, "eval/samples": len(got)}

    cfg = build_config(base_model, tokenizer_model, max_rows=len(train_rows),
                       completions_per_prompt=completions, prompt_groups_per_step=prompt_groups,
                       metrics_file=str(ROOT / "results" / f"rft_{env}_metrics.jsonl"))
    factory = functools.partial(make_rollout_fn, renderer_name=renderer_name)
    evaluation_fn = make_heldout_evaluation(
        held_out, samples_per_row=cfg.completions_per_prompt,
        metrics_fn=eval_metrics, max_concurrency=cfg.max_concurrency_rollout_sample,
    ) if held_out else None
    result = async_rl_loop_serverless.main(
        cfg,
        rollout_fn_factory=factory,
        dynamic_filter_fn=has_reward_variance,
        evaluation_fn=evaluation_fn,
        evaluation_interval=5,
        rows=train_rows,
    )
    # Persist the result (checkpoint / sampler-snapshot / session ids) so the
    # trained adapter can be promoted or re-evaluated later.
    try:
        out = ROOT / "results" / f"rft_{env}_result.json"
        out.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"[{env}] result -> {out}")
    except Exception as e:  # noqa: BLE001
        print(f"[{env}] could not write result json: {e}")
    return result


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def smoke() -> int:
    """Local dry run: prove rows load and the reward path works. No network, no spend."""
    print("[smoke] loading panel + rows (no API calls) ...")
    panel = core.load_panel(PANEL_PATH)
    for which in ("clean", "leaky", "holdout"):
        rows = load_rows(which)
        print(f"[smoke] {which:7s}: {len(rows)} rows; first gt = {rows[0]['ground_truth']}")

    gt = load_rows("clean")[0]["ground_truth"]
    good = 'Winners tend to be profitable and cheap.\n{"roe": 2, "ps": -1, "de": -1}'
    rg = reward_for_completion(good, gt, panel)
    print(f"[smoke] good answer   -> reward={rg['reward']:.4f} "
          f"spread={rg['spread']} vs v1={rg['base_spread']} parse_error={rg['parse_error']}")

    bad = "I think tech stocks will go up."
    rb = reward_for_completion(bad, gt, panel)
    print(f"[smoke] bad answer    -> reward={rb['reward']:.4f} parse_error={rb['parse_error']!r}")

    msgs = prompts.messages(no_think=True)
    print(f"[smoke] prompt: {len(msgs)} message(s); roles={[m.get('role') for m in msgs]}")

    ok = (rb["reward"] == 0.0 and rb["parse_error"] is not None)
    print("[smoke] PASS" if ok else "[smoke] CHECK: malformed answer did not score 0.0")
    return 0 if ok else 1


def check_wiring(base_model: str, tokenizer_model: str) -> int:
    """No-spend wiring check: import cookbook symbols, build Config + rows +
    heldout eval_fn, confirm make_rollout_fn is callable. Creates NO training
    client and NO sampler, so it never bills."""
    import inspect

    _ensure_cookbook_on_path()
    from training.recipes.experiment.async_rl_loop_serverless import main as sl_main
    from training.utils.rl.heldout import make_heldout_evaluation
    from training.utils.rl.losses import has_reward_variance
    from training.utils.rl.rollout import single_turn_renderer_rollout  # noqa: F401
    print(f"[wiring] cookbook at {COOKBOOK} (pin {COOKBOOK_PIN[:12]}); imports OK")

    rows = load_rows("clean")
    cfg = build_config(base_model, tokenizer_model, max_rows=len(rows) - 8)
    print(f"[wiring] Config: base={cfg.base_model} tok={cfg.tokenizer_model} "
          f"N={cfg.completions_per_prompt} groups/step={cfg.prompt_groups_per_step} "
          f"lora_rank={cfg.lora_rank} max_rows={cfg.max_rows} max_seq_len={cfg.max_seq_len}")

    eval_fn = make_heldout_evaluation(
        rows[:8], samples_per_row=cfg.completions_per_prompt,
        metrics_fn=lambda _r, runs: {"eval/mean_reward": 0.0},
        max_concurrency=cfg.max_concurrency_rollout_sample,
    )
    print(f"[wiring] heldout evaluation_fn over 8 rows: callable={callable(eval_fn)}")
    print(f"[wiring] make_rollout_fn{inspect.signature(make_rollout_fn)} ready "
          f"(uses single_turn_renderer_rollout)")
    print(f"[wiring] serverless main callable={callable(sl_main)}; "
          f"dynamic_filter={has_reward_variance.__name__}")
    print("[wiring] PASS — all wired. No training client/sampler created; nothing billed.")
    return 0


def cli() -> int:
    ap = argparse.ArgumentParser(description="finagic serverless RFT (GRPO) port")
    ap.add_argument("--smoke", action="store_true", help="local reward dry test; no API, no spend")
    ap.add_argument("--check-wiring", action="store_true",
                    help="import cookbook + build Config/rows/eval_fn; no spend")
    ap.add_argument("--train", choices=["clean", "leaky"], help="run serverless GRPO (BILLABLE)")
    ap.add_argument("--i-will-pay", action="store_true", help="required to actually train")
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--tokenizer-model", default=TOKENIZER_DEFAULT)
    ap.add_argument("--renderer-name", default=RENDERER_DEFAULT)
    ap.add_argument("--max-rows", type=int, default=85)
    ap.add_argument("--eval-rows", type=int, default=8)
    ap.add_argument("--completions", type=int, default=8)
    ap.add_argument("--prompt-groups", type=int, default=8)
    args = ap.parse_args()
    if args.smoke:
        return smoke()
    if args.check_wiring:
        return check_wiring(args.base_model, args.tokenizer_model)
    if args.train:
        train(args.train, args.base_model, args.tokenizer_model,
              eval_rows=args.eval_rows, max_rows=args.max_rows,
              completions=args.completions, prompt_groups=args.prompt_groups,
              renderer_name=args.renderer_name, i_will_pay=args.i_will_pay)
        return 0
    print("Nothing to do. Use --smoke (local reward), --check-wiring (no spend), "
          "or --train {clean,leaky} --i-will-pay (billable).")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
