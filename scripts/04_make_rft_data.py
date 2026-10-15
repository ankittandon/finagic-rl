"""Step 4: write the RFT datasets (JSONL, Eval Protocol row format).

  rft/train_clean.jsonl    one row per 12-month window in the training years, graded clean
  rft/train_leaky.jsonl    the same windows, graded in the leaky environment
  rft/holdout_clean.jsonl  windows in the holdout years, graded clean (the evaluation dataset)

Every row has the same prompt; only the grading window differs. That makes the job a pure
test of the environment: the clean and leaky runs see identical prompts and differ only in
what the reward is computed from.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finagic import core, prompts  # noqa: E402
from finagic.build import shift_months  # noqa: E402


def starts(lo: int, hi: int, step: int = 1):
    m = lo
    while m <= hi:
        yield m
        m = shift_months(m, step)


def write(path: Path, env: str, months, no_think: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for m in months:
            row = {"messages": prompts.messages(no_think),
                   "ground_truth": {"env": env, "start": int(m), "n_months": 12}}
            f.write(json.dumps(row) + "\n")
            n += 1
    print(f"wrote {path.relative_to(ROOT)} ({n} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-think", action="store_true", help="append /no_think (Qwen3 models)")
    args = ap.parse_args()
    tr_lo, tr_hi = core.SPLITS["train"]
    ho_lo, ho_hi = core.SPLITS["holdout"]
    # a training window's 12 decision months must end inside the training years
    train_starts = list(starts(tr_lo, shift_months(tr_hi, -11)))
    hold_starts = list(starts(ho_lo, shift_months(ho_hi, -11)))
    write(ROOT / "rft" / "train_clean.jsonl", "clean", train_starts, args.no_think)
    write(ROOT / "rft" / "train_leaky.jsonl", "leaky", train_starts, args.no_think)
    write(ROOT / "rft" / "holdout_clean.jsonl", "clean", hold_starts, args.no_think)


if __name__ == "__main__":
    main()
