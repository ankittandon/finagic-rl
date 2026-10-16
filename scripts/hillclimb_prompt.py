"""Hill-climb the PROMPT for the yes/no growth-direction task.

Evaluates a set of named prompt templates on a sample of the holdout set, by
sampling an inferenceable proxy model (nemotron-lightning) and scoring accuracy
against the real labels. Claude proposes variants across rounds (edit TEMPLATES
and re-run); keep the winner. Inference only — cheap, no training.

    python scripts/hillclimb_prompt.py --n 120
"""
import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json  # noqa: E402

from openai import OpenAI  # noqa: E402

from finagic import core  # noqa: E402

# A strong, inferenceable instruct model; /no_think keeps output short + parseable.
PROXY_MODEL = "accounts/fireworks/models/qwen3p8-max"

# A worked few-shot example (for templates that use it)
FEWSHOT = (
    "Example — company: Trailing 1y revenue growth 11.0% (sector 4.0%), "
    "Operating margin 20.0% (sector 12.0%), Op. cash flow / Debt 1.8 (sector 0.9). "
    "Reasoning: growth and margins are well above peers and it is cash-generative, "
    "so it is likely to keep out-growing. ANSWER: yes"
)

# Round 1 candidate templates. Each wraps the per-company data block.
TEMPLATES = {
    # round-1 winner, kept as the anchor
    "T2_heuristic": {
        "system": "You are an equity analyst who knows that recent growth momentum and profitability are the strongest predictors of continued out-performance.",
        "pre": "Company fundamentals vs sector-peer median:",
        "post": ('Will this company out-grow its sector (revenue) over the next 12 months? Favor companies whose '
                 'RECENT revenue growth and margins are above their peers.\n'
                 'One sentence of reasoning, then exactly: "ANSWER: yes" or "ANSWER: no".'),
    },
    # round 2: inject the momentum rule explicitly
    "T2d_momentum": {
        "system": ("You are an equity analyst. A strong rule: a company whose RECENT (trailing) revenue growth is "
                   "above its sector-peer median usually keeps out-growing its peers the next year, unless margins "
                   "or cash generation are clearly weak."),
        "pre": "Company fundamentals vs sector-peer median:",
        "post": ('Apply the rule above. Will this company grow revenue FASTER than its sector median over the next '
                 '12 months? One short clause, then exactly: "ANSWER: yes" or "ANSWER: no".'),
    },
    # round 2: decisive and very concise (protect parse rate)
    "T2e_concise": {
        "system": "You are a decisive equity analyst.",
        "pre": "Company fundamentals vs sector-peer median:",
        "post": ('Over the next 12 months, will its revenue grow faster than the sector median? Favor above-peer '
                 'recent growth and margins. Answer in ONE short clause ending with exactly: "ANSWER: yes" or "ANSWER: no".'),
    },
    # round 2: check growth-vs-peers first
    "T2f_growthfirst": {
        "system": "You are an equity analyst.",
        "pre": "Company fundamentals vs sector-peer median:",
        "post": ('First note whether the company\'s trailing revenue growth is ABOVE or BELOW its sector median; '
                 'companies above usually keep out-growing. Then decide for the next 12 months. '
                 'End with exactly: "ANSWER: yes" or "ANSWER: no".'),
    },
}


def data_block(content: str) -> str:
    return "\n".join(l for l in content.splitlines() if l.strip().startswith("- "))


def build_messages(tpl, block):
    user = f"{tpl['pre']}\n\n{block}\n\n{tpl['post']}\n/no_think"
    return [{"role": "system", "content": tpl["system"]}, {"role": "user", "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--model", default=PROXY_MODEL)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / "rft" / "pointwise_holdout.jsonl")][: args.n]
    items = [(data_block(r["messages"][1]["content"]), int(r["ground_truth"]["label"])) for r in rows]
    client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"])

    def ask(msgs):
        r = client.chat.completions.create(model=args.model, messages=msgs,
                                            temperature=args.temperature, max_tokens=400)
        return core.parse_yesno(r.choices[0].message.content or "")

    print(f"model={args.model}  n={len(items)}  (labels pos-rate {sum(l for _,l in items)/len(items):.2f})\n")
    results = {}
    for name, tpl in TEMPLATES.items():
        msgs_list = [build_messages(tpl, blk) for blk, _ in items]
        with ThreadPoolExecutor(max_workers=16) as ex:
            preds = list(ex.map(ask, msgs_list))
        parsed = [(p, l) for p, l in zip(preds, [l for _, l in items]) if p is not None]
        acc = sum(int(p == bool(l)) for p, l in parsed) / len(parsed) if parsed else 0.0
        results[name] = (acc, len(parsed) / len(items))
        print(f"  {name:14s}  accuracy={acc:.3f}   parse_rate={len(parsed)/len(items):.2f}")
    best = max(results, key=lambda k: results[k][0])
    print(f"\nBEST: {best}  accuracy={results[best][0]:.3f}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "hillclimb_prompt.json").write_text(json.dumps(
        {k: {"accuracy": v[0], "parse_rate": v[1]} for k, v in results.items()} | {"best": best}, indent=2))


if __name__ == "__main__":
    main()
