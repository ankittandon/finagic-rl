"""Build a REASONING-TRACE SFT set by teacher distillation + rejection sampling.

The plain-label SFT (ANSWER: yes/no with no reasoning) overfits and collapses to
chance. Fireworks' documented fix: don't train on bare labels — distill *reasoning
traces* from a stronger model and keep only the traces that reach the correct
answer (rejection sampling). The student then learns a process, not a label table.

Teacher = qwen3p8-max (reaches ~62% here). For each neutral-prompt train row we
sample a full chain-of-thought, parse its ANSWER, and keep the trace iff it matches
the ground-truth label. Writes rft/neutral_sft_reason.jsonl.

    python scripts/build_reason_sft.py --max 1600
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finagic import core  # noqa: E402
from openai import OpenAI  # noqa: E402

TEACHER = "accounts/fireworks/models/qwen3p8-max"


def clean_trace(text: str) -> str | None:
    """Strip <think> wrappers, keep visible reasoning, require a final ANSWER line."""
    if not text:
        return None
    # unwrap reasoning tags but keep their content
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    ans = core.parse_yesno(text)
    if ans is None:
        return None
    # normalize the ending to exactly one ANSWER line
    body = re.split(r"ANSWER\s*:", text, flags=re.IGNORECASE)[0].strip()
    if not body:
        return None
    body = body[-1800:]  # cap trace length
    return f"{body}\n\nANSWER: {'yes' if ans else 'no'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1600)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / "rft" / "neutral_train.jsonl")][: args.max]
    client = OpenAI(base_url="https://api.fireworks.ai/inference/v1",
                    api_key=os.environ["FIREWORKS_API_KEY"])

    def sample(row):
        msgs = row["messages"]
        label = int(row["ground_truth"]["label"])
        # Teacher prompt: force a SHORT, parseable rationale (/no_think, as the
        # hill-climb used) so reasoning doesn't truncate before the ANSWER line.
        tmsgs = [msgs[0], {"role": "user", "content":
                 msgs[1]["content"].replace("Think it through, then end",
                 "Give one short sentence of reasoning, then end") + "\n/no_think"}]
        try:
            r = client.chat.completions.create(model=TEACHER, messages=tmsgs,
                                               temperature=args.temperature, max_tokens=512)
            trace = clean_trace(r.choices[0].message.content or "")
        except Exception:
            return None
        if trace is None:
            return None
        pred = core.parse_yesno(trace)
        if pred is None or int(pred) != label:   # rejection sampling: keep only correct traces
            return None
        # train the student on the CLEAN neutral prompt (no /no_think) + the trace
        return (label, {"messages": msgs + [{"role": "assistant", "content": trace}]})

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        kept = [r for r in ex.map(sample, rows) if r is not None]

    # balance classes: downsample the majority so the student can't just say "no"
    import random as _rnd
    _rnd.seed(0)
    yes = [r for lab, r in kept if lab == 1]
    no = [r for lab, r in kept if lab == 0]
    n = min(len(yes), len(no))
    bal = yes[:n] + no[:n]
    _rnd.shuffle(bal)

    out = ROOT / "rft" / "neutral_sft_reason.jsonl"
    with out.open("w") as f:
        for r in bal:
            f.write(json.dumps(r) + "\n")
    print(f"teacher={TEACHER}")
    print(f"sampled {len(rows)} -> kept {len(kept)} correct "
          f"({len(kept)/len(rows):.1%}); balanced to {len(bal)} ({n} yes / {n} no)")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
