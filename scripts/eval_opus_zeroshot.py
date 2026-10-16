"""Zero-shot frontier-model rung: score Claude Opus on the SAME pointwise task.

No training, no hint tuning — give Opus the holdout prompt cold and check its
yes/no answers against the real labels. Answers the question "do you even need
RL — can't a big general model just do this out of the box?"

Defaults to the NEUTRAL holdout (the fair, no-hint test), same 400 rows every
other rung is scored on. Pass --holdout rft/pointwise_holdout.jsonl for the
hinted prompt instead.

    ANTHROPIC_API_KEY=... python scripts/eval_opus_zeroshot.py --n 400

Reads ANTHROPIC_API_KEY from the environment (keep it in the gitignored .env).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finagic import core  # noqa: E402
import anthropic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--holdout", default="rft/neutral_holdout.jsonl")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set — add it to .env (it is gitignored) and re-run.")

    rows = [json.loads(l) for l in open(ROOT / args.holdout)][: args.n]
    # org-scoped keys must name a workspace via header; workspace-scoped keys don't.
    ckw = {}
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if ws:
        ckw["default_headers"] = {"anthropic-workspace-id": ws}
    client = anthropic.Anthropic(**ckw)

    def ask(row):
        sys_msg = row["messages"][0]["content"]
        user_msg = row["messages"][1]["content"]
        label = int(row["ground_truth"]["label"])
        try:
            r = client.messages.create(
                model=args.model, max_tokens=args.max_tokens,
                system=sys_msg, messages=[{"role": "user", "content": user_msg}])
            text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        except Exception as e:  # noqa: BLE001
            return (None, label, f"ERR:{type(e).__name__}")
        return (core.parse_yesno(text), label, text[:0])

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(ask, rows))

    n = len(res)
    parsed = [(p, l) for p, l, _ in res if p is not None]
    err = sum(1 for _, _, note in res if note.startswith("ERR:"))
    correct = sum(1 for p, l in parsed if int(p) == l)
    acc = correct / n if n else 0.0                      # unparsed counts as wrong (same as RL eval/accuracy)
    acc_parsed = correct / len(parsed) if parsed else 0.0
    out = {"model": args.model, "holdout": args.holdout, "n": n,
           "accuracy": round(acc, 4), "acc_parsed": round(acc_parsed, 4),
           "parse_rate": round(len(parsed) / n, 4) if n else 0.0, "api_errors": err}
    print(json.dumps(out, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    tag = "neutral" if "neutral" in args.holdout else "hinted"
    (ROOT / "results" / f"opus_zeroshot_{tag}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
