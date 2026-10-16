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
    # pointwise yes/no growth-direction task (per-company prompts, built by build_pointwise.py)
    "pt_train": ROOT / "rft" / "pointwise_train.jsonl",
    "pt_holdout": ROOT / "rft" / "pointwise_holdout.jsonl",
}
# Let the pointwise train/holdout files be overridden (e.g. the neutral-prompt
# variant) without touching the default hinted datasets.
for _k, _ev in (("pt_train", "FINAGIC_PT_TRAIN"), ("pt_holdout", "FINAGIC_PT_HOLDOUT")):
    if os.environ.get(_ev):
        DATASETS[_k] = Path(os.environ[_ev])

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


def reward_for_completion(completion_text: str, ground_truth: dict, panel=None, task: str = "formula") -> dict:
    """completion string -> reward, for either task.

    pointwise: yes/no growth-direction (correct=1.0, well-formed-wrong=0.1, else 0).
    formula:   finagic rank-IC growth reward over a window (0.5 = no skill).
    Pure/local: no API calls, no cost.
    """
    if task == "pointwise":
        g = core.grade_pointwise(completion_text, int(ground_truth["label"]))
        return {"reward": float(g["reward"]), "pred": g["pred"], "correct": g["correct"],
                "parse_error": None if g["pred"] is not None else "no yes/no answer"}
    if panel is None:
        panel = core.load_panel(PANEL_PATH)
    formula, err = core.parse_formula(completion_text)
    if formula is None:
        return {"reward": 0.0, "ic": None, "parse_error": err}
    r = core.reward_for_growth(
        panel, ground_truth["env"], int(ground_truth["start"]),
        formula, int(ground_truth.get("n_months", 12)),
    )
    return {"reward": float(r["reward"]), "ic": r.get("ic"), "parse_error": None}


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


def make_rollout_fn(setup, *, renderer_name: str = "", task: str = "formula"):
    """One sampled completion per call, rendered and graded by the task reward.

    single_turn_renderer_rollout handles sampling, tokenization and GRPO datum
    assembly; we pass the prompt (already in row["messages"]) and the scalar reward.
    """
    import asyncio

    from training.renderer import get_text_content
    from training.utils import build_renderer, prepare_sampling_messages
    from training.utils.rl.rollout import single_turn_renderer_rollout

    renderer = build_renderer(
        setup.tokenizer, setup.tokenizer_id, renderer_name, load_image_processor=False
    )
    sampler = setup.sampler              # borrowed from the recipe; do NOT close
    panel = None if task == "pointwise" else core.load_panel(PANEL_PATH)

    async def grade(row: dict, message, _parse_success: bool) -> float:
        text = get_text_content(message)
        gt = row["ground_truth"]
        return await asyncio.to_thread(lambda: reward_for_completion(text, gt, panel, task)["reward"])

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
                 learning_rate: float = 1e-4, metrics_file: str | None = None,
                 seed: int | None = None, eps_clip_high: float | None = None,
                 init_from_checkpoint: str | None = None):
    """Build the serverless GRPO Config for one experiment."""
    _ensure_cookbook_on_path()
    from training.recipes.experiment.async_rl_loop_serverless import Config
    extra = {} if seed is None else {"seed": seed}
    if eps_clip_high is not None:   # DAPO-style clip-higher to resist entropy collapse
        extra["eps_clip_high"] = eps_clip_high
    if init_from_checkpoint is not None:   # evaluate/continue from an existing adapter
        extra["init_from_checkpoint"] = init_from_checkpoint
    return Config(
        **extra,
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


def train(env: str, base_model: str, tokenizer_model: str, *, task: str = "formula",
          eval_rows: int = 8, max_rows: int = 85, completions: int = 8, prompt_groups: int = 8,
          renderer_name: str = RENDERER_DEFAULT, seed: int | None = None,
          learning_rate: float | None = None, eps_clip_high: float | None = None,
          init_from_checkpoint: str | None = None, i_will_pay: bool = False):
    """Run serverless GRPO. BILLABLE — guarded. task='pointwise' trains the yes/no
    growth-direction task (per-company prompts, dense reward, stabilized); task=
    'formula' trains the rank-IC formula task on the clean holdout."""
    if not i_will_pay:
        raise SystemExit("Refusing to start a billable training run without --i-will-pay.")
    _ensure_cookbook_on_path()
    import functools

    from training.recipes.experiment import async_rl_loop_serverless
    from training.utils.rl.heldout import make_heldout_evaluation
    from training.utils.rl.losses import has_reward_variance

    if task == "pointwise":
        train_rows = load_rows("pt_train")[:max_rows]
        held_out = load_rows("pt_holdout")[:eval_rows]
        tag = "pt"
        if learning_rate is None:
            learning_rate = 2e-5          # lower LR — stabilize vs the formula task
        if eps_clip_high is None:
            eps_clip_high = 0.28          # DAPO-style clip-higher
    else:
        train_rows = load_rows(env)[:max_rows]
        held_out = load_rows("holdout")[:eval_rows]
        tag = env
        if learning_rate is None:
            learning_rate = 1e-4
    if not train_rows:
        raise SystemExit(f"no training rows ({task})")

    def eval_metrics(_rows, runs):
        got = [r.segments[0].reward for rr in runs for r in rr if r is not None]
        exp = sum(len(rr) for rr in runs)
        out = {"eval/mean_reward": (sum(got) / exp) if exp else 0.0, "eval/samples": len(got)}
        if task == "pointwise" and got:   # exact accuracy from the reward (1=correct, 0.1=wrong, 0=unparsed)
            correct = sum(1 for x in got if x >= 0.999)
            parsed = sum(1 for x in got if x > 0.0)
            out["eval/accuracy"] = correct / len(got)
            out["eval/acc_parsed"] = (correct / parsed) if parsed else 0.0
            out["eval/parse_rate"] = parsed / len(got)
        return out

    suffix = f"_s{seed}" if seed is not None else ""
    cfg = build_config(base_model, tokenizer_model, max_rows=len(train_rows),
                       completions_per_prompt=completions, prompt_groups_per_step=prompt_groups,
                       learning_rate=learning_rate, eps_clip_high=eps_clip_high,
                       init_from_checkpoint=init_from_checkpoint,
                       metrics_file=str(ROOT / "results" / f"rft_{tag}{suffix}_metrics.jsonl"),
                       seed=seed)
    factory = functools.partial(make_rollout_fn, renderer_name=renderer_name, task=task)
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
    try:
        out = ROOT / "results" / f"rft_{tag}{suffix}_result.json"
        out.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"[{tag}] result -> {out}")
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] could not write result json: {e}")
    return result


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def smoke(task: str = "formula") -> int:
    """Local dry run: prove rows load and the reward path works. No network, no spend."""
    if task == "pointwise":
        print("[smoke] pointwise: loading rows (no API calls) ...")
        for which in ("pt_train", "pt_holdout"):
            rows = load_rows(which)
            print(f"[smoke] {which:11s}: {len(rows)} rows; first label = {rows[0]['ground_truth']['label']}")
        row = load_rows("pt_train")[0]
        gt = row["ground_truth"]
        yes_ans = "yes" if gt["label"] == 1 else "no"
        no_ans = "no" if gt["label"] == 1 else "yes"
        rc = reward_for_completion(f"Strong momentum.\nANSWER: {yes_ans}", gt, None, "pointwise")
        rw = reward_for_completion(f"Looks weak.\nANSWER: {no_ans}", gt, None, "pointwise")
        rb = reward_for_completion("I am not sure about this company.", gt, None, "pointwise")
        print(f"[smoke] correct answer -> reward={rc['reward']:.2f}  pred={rc['pred']} correct={rc['correct']}")
        print(f"[smoke] wrong answer   -> reward={rw['reward']:.2f}  (format floor)")
        print(f"[smoke] no answer      -> reward={rb['reward']:.2f}  parse_error={rb['parse_error']!r}")
        print(f"[smoke] sample prompt (user, truncated):\n  {row['messages'][1]['content'][:320]} ...")
        ok = (rc["reward"] == 1.0 and 0.0 < rw["reward"] < 1.0 and rb["reward"] == 0.0)
        print("[smoke] PASS" if ok else "[smoke] CHECK: pointwise reward did not behave as expected")
        return 0 if ok else 1

    print("[smoke] loading panel + rows (no API calls) ...")
    panel = core.load_panel(PANEL_PATH)
    for which in ("clean", "leaky", "holdout"):
        rows = load_rows(which)
        print(f"[smoke] {which:7s}: {len(rows)} rows; first gt = {rows[0]['ground_truth']}")

    gt = load_rows("clean")[0]["ground_truth"]
    # momentum is now residualized OUT of the target, so a pure-momentum formula should
    # score ~0 IC; a non-momentum (quality/value) formula tests the residual signal.
    mom = 'Momentum.\n{"tg_rev": 2, "tg_ocf": 1, "tg_gp": 1}'
    rm = reward_for_completion(mom, gt, panel)
    qual = 'Quality/value.\n{"op_margin": 2, "roe": 1, "pb": -1}'
    rq = reward_for_completion(qual, gt, panel)
    bad = "I think tech stocks will go up."
    rb = reward_for_completion(bad, gt, panel)
    print(f"[smoke] momentum formula -> reward={rm['reward']:.4f}  rank-IC={rm['ic']}  (expect ~0 now)")
    print(f"[smoke] quality/value    -> reward={rq['reward']:.4f}  rank-IC={rq['ic']}")
    print(f"[smoke] non-answer       -> reward={rb['reward']:.4f}  parse_error={rb['parse_error']!r}")

    msgs = prompts.messages(no_think=True)
    print(f"[smoke] prompt: {len(msgs)} message(s); roles={[m.get('role') for m in msgs]}")

    ok = (rb["reward"] == 0.0 and rb["parse_error"] is not None
          and rm["ic"] is not None and rq["ic"] is not None)
    print("[smoke] PASS" if ok else "[smoke] CHECK: reward did not compute as expected")
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
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--task", choices=["formula", "pointwise"], default="formula")
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--init-from-checkpoint", default=None,
                    help="evaluate/continue from an existing adapter checkpoint")
    args = ap.parse_args()
    if args.smoke:
        return smoke(args.task)
    if args.check_wiring:
        return check_wiring(args.base_model, args.tokenizer_model)
    if args.train:
        train(args.train, args.base_model, args.tokenizer_model, task=args.task,
              eval_rows=args.eval_rows, max_rows=args.max_rows,
              completions=args.completions, prompt_groups=args.prompt_groups,
              renderer_name=args.renderer_name, seed=args.seed,
              learning_rate=args.learning_rate,
              init_from_checkpoint=args.init_from_checkpoint, i_will_pay=args.i_will_pay)
        return 0
    print("Nothing to do. Use --smoke (local reward), --check-wiring (no spend), "
          "or --train {clean,leaky} --i-will-pay (billable).")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
