"""Step 6: sample formulas from a Fireworks model and backtest them.

Base model, before any training (zero-shot and best-of-N):
    python scripts/06_sample_model.py --label base --n 16

A tuned model after RFT (once it is deployed; see the runbook):
    python scripts/06_sample_model.py --label rft-clean --model accounts/<acct>/models/<output-model> --n 16

Best-of-N picks the sample with the best spread on the clean TRAINING years and reports its
holdout spread, so the selection never looks at the holdout. Also records latency and tokens.
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finagic import core, prompts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default=os.environ.get("FINAGIC_EVAL_MODEL", "accounts/fireworks/models/qwen3-8b"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (from fireworks.ai/pricing)")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    args = ap.parse_args()
    model = args.model.removeprefix("fireworks_ai/")

    client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"])
    msgs = prompts.messages(args.no_think)

    def one(i):
        t0 = time.time()
        r = client.chat.completions.create(model=model, messages=msgs, temperature=args.temperature,
                                           max_tokens=args.max_tokens)
        dt = time.time() - t0
        text = r.choices[0].message.content or ""
        u = r.usage
        return {"i": i, "latency_s": dt, "text": text,
                "in_tok": getattr(u, "prompt_tokens", 0), "out_tok": getattr(u, "completion_tokens", 0)}

    print(f"sampling {args.n} answers from {model} ...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        samples = list(ex.map(one, range(args.n)))

    panel = core.load_panel(ROOT / "evaluator" / "data" / "panel.npz")
    idx = {s: core.split_idx(panel, s) for s in ("train", "val", "holdout")}
    for s in samples:
        f, err = core.parse_formula(s["text"])
        s["formula"], s["parse_error"] = f, err
        if f:
            for split, ix in idx.items():
                s[f"{split}_clean"] = core.evaluate(panel, "clean", ix, f)["spread"]
            s["practice_leaky"] = core.evaluate(panel, "leaky", idx["train"], f)["spread"]

    ok = [s for s in samples if s["formula"]]
    out = {"label": args.label, "model": model, "n": args.n, "parsed": len(ok),
           "latency_p50_s": statistics.median(s["latency_s"] for s in samples),
           "latency_p95_s": sorted(s["latency_s"] for s in samples)[max(0, int(0.95 * len(samples)) - 1)],
           "in_tokens": sum(s["in_tok"] for s in samples), "out_tokens": sum(s["out_tok"] for s in samples)}
    if args.price_in is not None and args.price_out is not None:
        out["cost_usd"] = out["in_tokens"] / 1e6 * args.price_in + out["out_tokens"] / 1e6 * args.price_out
    if ok:
        out["mean_holdout_clean"] = statistics.mean(s["holdout_clean"] for s in ok)
        out["mean_train_clean"] = statistics.mean(s["train_clean"] for s in ok)
        best = max(ok, key=lambda s: s["train_clean"])
        out["best_of_n"] = {k: best[k] for k in ("train_clean", "val_clean", "holdout_clean", "formula")}
    out["samples"] = samples
    (ROOT / "results").mkdir(exist_ok=True)
    path = ROOT / "results" / f"samples_{args.label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=2))
    print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
